'''
Collection of boilerplate and utility functions for PyTorch's processing pipeline.
Many are adapted and expanded from https://github.com/mrdbourke/pytorch-deep-learning/
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary
import torchmetrics
import timm

import numpy as np

import requests
from datetime import datetime
import os
from pathlib import Path
import zipfile

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from typing import Callable



#### Misc Util Functions ####

def set_seeds(seed: int = 42):
    '''Set both torch and torch.cuda seeds.'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




#### Training & Testing Functions ####

def train_combinations(combinations: dict[str, tuple[str, str, str, int, str]],
                       model_factories: dict[str, Callable[[], nn.Module]], train_dataloaders: dict[str, DataLoader],
                       optimiser_factories: dict[str, Callable[[nn.Module], torch.optim.Optimizer]],
                       test_dataloader: DataLoader, loss_fn: nn.Module, metric_name_and_fn: tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
                       reset_seed: int = 42, device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu', show_progress_bar = True):
    '''Run a series of modelling tasks by defining combinations of models, dataloaders, optimisers and epochs, as well as an optional previously-fit combination
    to start from (e.g. for a combination which is the same as a previous one but with more epochs or different training data).
    
    Models' state_dicts are saved in ./models, and evaluation metrics in ./runs (and also printed throughout).

    :param combinations: The experiment program, a dictionary of VERY SHORT keys (used as model naming prefixes) mapped to tuples containing
        dictionary keys to the model ingredients: (model key, train dataloader key, optimiser key, epochs, OPTIONAL KEY OF PREVIOUS COMBINATION TO START FROM)
    :param model_factories: Named model-producing functions
    :param train_dataloaders: Named dataloaders for training data
    :param optimiser_factories: Named optimiser-generating functions (e.g. dict(Adam001 = lambda m: torch.optim.Adam(m.parameters(), lr = 0.001)))
    :param test_dataloader: The dataloader for testing data
    :param loss_fn: A loss function taking in only prediction and target tensors
    :param metric_name_and_fn: A tuple of metric name and function taking in only prediction and target tensors
    :param reset_seed: A seed to re-impose (on both torch and torch.cuda) before every combination is executed (or None to not do so)
    :param device: A target device to compute on (e.g. 'cuda' or 'cpu')
    :param show_progress_bar: Show a progress bar for the experiment loop, their nested epoch loops and each of the nested training and testing steps batch loops
    '''
    # Well worth checking labelling issues before the long processing
    assert all(len(comb) == 5 for comb in combinations.values()), 'Some combinations are not 5-tuples; they should be of the form (model key, train dataloader key, optimiser key, epochs, None or previous combination key)'
    ms, ds, os, es, bcs = zip(*combinations.values()) # bcs stands for base combinations
    for keys, ingredients, param_name in [(ms, model_factories, 'model_factories'), (ds, train_dataloaders, 'train_dataloaders'), (os, optimiser_factories, 'optimiser_factories')]:
        assert not (set_diff := set(keys).difference(ingredients.keys())), f'Combination ingredient(s) {set_diff} not present in the {param_name} dictionary keys'
    assert not (set_diff := set(bcs).difference([None]).difference(combinations.keys())), f'Base model key(s) {set_diff} not present in the combination dictionary keys'
    combs_order = {k: i for i, k in enumerate(combinations.keys())}
    for comb_with_bc, bc, stated_m in {(k, vs[-1], vs[0]) for k, vs in combinations.items() if vs[-1] is not None}:
        assert combinations[bc][0] == stated_m, f'The stated model for combination {comb_with_bc} ({stated_m}) does not match the one of its stated base combination {bc} ({combinations[bc][0]})'
        assert combs_order[bc] < combs_order[comb_with_bc], f'Combination {comb_with_bc} (#{combs_order[comb_with_bc]}) requires combination {bc} (#{combs_order[bc]}) but occurs before it in the combination order'

    saved_models = dict()
    for experiment_number, combination_key in tqdm(combinations.keys(), desc = 'Modelling combinations', disable = not show_progress_bar):
        model_key, train_data_key, optimiser_key, epochs, base_comb_key = combinations[combination_key]
        print(f'[INFO] Experiment number: {experiment_number}')
        print(f'[INFO] Model: {model_key}')
        print(f'[INFO] DataLoader: {train_data_key}')
        print(f'[INFO] Number of epochs: {epochs}')  
        print(f'[INFO] Base model to build on: {base_comb_key}')  

        model = model_factories[model_key]()
        if base_comb_key is not None: model.load_state_dict(torch.load(saved_models[base_comb_key]))

        if reset_seed: set_seeds(reset_seed)
        
        fit(model = model, train_dataloader = train_dataloaders[train_data_key], test_dataloader = test_dataloader,
            optimiser = optimiser_factories[optimiser_key](model), loss_fn = loss_fn, metric_name_and_fn = metric_name_and_fn,
            epochs = epochs, device = device, show_progress_bar = show_progress_bar,
            model_name = f'Combination {experiment_number}: {combination_key} - {optimiser_key}',
            writer = tensorboard_writer(experiment_name = train_data_key, model_name = model_key, extra = f'{experiment_number}_{combination_key}_{optimiser_key}_{epochs}_epochs'))
        
        saved_models[combination_key] = save_model(model = model, target_dir = 'models', model_name = f'{experiment_number}_{combination_key}_{model_key}_{train_data_key}_{optimiser_key}_{epochs}_epochs.pth')
        print('-'*50 + '\n')


def fit(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, 
        optimiser: torch.optim.Optimizer, loss_fn: nn.Module, metric_name_and_fn: tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        epochs: int, writer: torch.utils.tensorboard.writer.SummaryWriter,
        device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu', show_progress_bar = True, model_name: str = None) -> dict[str, list]:
    '''Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step() functions for a number of epochs,
    training and testing the model in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    :param model: A PyTorch model to be trained and tested
    :param train_dataloader: A DataLoader instance for the model to be trained on
    :param test_dataloader: A DataLoader instance for the model to be tested on
    :param optimiser: A PyTorch optimizer to help minimize the loss function
    :param loss_fn: A loss function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param metric_name_and_fn: A tuple of metric name and function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param epochs: An integer indicating how many epochs to train for
    :param writer: A SummaryWriter() instance to log model results to (set to None otherwise). E.g. tensorboard_writer(experiment_name = ..., model_name = ..., extra = f'{experiment_number}_{combination_key}_{optimiser_key}_{epochs}_epochs')
    :param device: A target device to compute on (e.g. 'cuda' or 'cpu')
    :param show_progress_bar: Show a progress bar for the global epoch loop and each of the nested training and testing steps batch loops
    :param model_name: A label to display in the progress bar if shown
    :return: A dictionary of training and testing loss as well as training and testing accuracy metrics.d
        Each metric has a value in a list for each epoch: {train_loss: [...], train_metric: [...], test_loss: [...], test_metric: [...]}
    '''
    keys = ['train_loss', 'train_metric', 'test_loss', 'test_metric']
    results = {k : [] for k in keys}
    
    model.to(device)
    for epoch in tqdm(range(1, epochs + 1), desc = model_name, disable = not show_progress_bar):
        train_loss, train_metric = training_step(model = model, dataloader = train_dataloader, loss_fn = loss_fn, metric_fn = metric_name_and_fn[1], optimiser = optimiser, device = device, show_progress_bar = show_progress_bar, epoch = epoch)
        test_loss,  test_metric  = testing_step( model = model, dataloader = test_dataloader,  loss_fn = loss_fn, metric_fn = metric_name_and_fn[1],                        device = device, show_progress_bar = show_progress_bar, epoch = epoch)

        print(
          f'Epoch: {epoch} | '
          f'train_loss: {train_loss:.4f} | '
          f'train_metric: {train_metric:.4f} | '
          f'test_loss: {test_loss:.4f} | '
          f'test_metric: {test_metric:.4f}'
        )

        for k, v in zip(keys, [train_loss, train_metric, test_loss, test_metric]): results[k].append(v)

        if writer is not None:
            writer.add_scalars(main_tag = 'Loss', tag_scalar_dict = dict(train_loss = train_loss, test_loss = test_loss), global_step = epoch)
            writer.add_scalars(main_tag = metric_name_and_fn[0], tag_scalar_dict = dict(train_metric = train_metric, test_metric = test_metric), global_step = epoch)
            writer.add_graph(model = model, input_to_model = torch.randn(32, 3, 224, 224).to(device)) # pass an example input
    
    if writer is not None: writer.close()
    return results


def training_step(model: nn.Module, dataloader: DataLoader,
                  loss_fn: nn.Module, metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optimiser: torch.optim.Optimizer,
                  device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu', show_progress_bar = True, epoch: int = None) -> tuple[float, float]:
    '''Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step, metric calculation).

    :param model: A PyTorch model to be trained
    :param dataloader: A DataLoader instance for the model to be trained on
    :param loss_fn: A loss function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param metric_fn: A performance metric function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param optimizer: A PyTorch optimizer to help minimize the loss function
    :param device: A target device to compute on (e.g. 'cuda' or 'cpu')
    :param show_progress_bar: Show a progress bar for the training loop over batches
    :return: A tuple of training loss and training metric
    '''
    model.train()

    progress_bar = tqdm(enumerate(dataloader), desc = f'{"T" if epoch is None else f"Epoch {epoch} t"}raining batches', disable = not show_progress_bar)

    train_loss, train_metric = 0, 0
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimiser.zero_grad() # set_to_none is True by default
        loss.backward()
        optimiser.step()

        train_metric += metric_fn(y_pred, y).item()

        progress_bar.set_postfix(dict(train_loss = train_loss / (batch + 1), train_metric = train_metric / (batch + 1)))

    return train_loss / len(dataloader), train_metric / len(dataloader) # batch mean of the metrics 


def testing_step(model: nn.Module, dataloader: DataLoader, 
                 loss_fn: nn.Module, metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu', show_progress_bar = True, epoch: int = None) -> tuple[float, float]:
    '''Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to 'eval' mode and then performs a forward pass on a testing dataset.

    :param model: A PyTorch model to be tested
    :param dataloader: A DataLoader instance for the model to be tested on
    :param loss_fn: A loss function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param metric_fn: A performance metric function taking in only prediction and target tensors and returning a tensor (.item() is used where appropriate)
    :param device: A target device to compute on (e.g. 'cuda' or 'cpu')
    :param show_progress_bar: Show a progress bar for the testing loop over batches
    :return: A tuple of testing loss and testing metric
    '''
    model.eval()

    progress_bar = tqdm(enumerate(dataloader), desc = f'{"T" if epoch is None else f"Epoch {epoch} t"}esting batches', disable = not show_progress_bar)

    test_loss, test_metric = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            test_loss   += loss_fn(  y_pred, y).item()
            test_metric += metric_fn(y_pred, y).item()

            progress_bar.set_postfix(dict(test_loss = test_loss / (batch + 1), test_metric = test_metric / (batch + 1)))

    return test_loss / len(dataloader), test_metric / len(dataloader) # batch mean of the metrics 




#### I/O Functions ####

def save_model(model: nn.Module, target_dir: str, model_name: str):
    '''Saves a PyTorch model to a target directory.

    :param model: A target PyTorch model to save
    :param target_dir: A directory for saving the model to
    :param model_name: A filename for the saved model; should include either '.pth' or '.pt' as the file extension
    '''
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents = True, exist_ok = True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model_name should end with ".pt" or ".pth"'
    model_save_path = target_dir_path / model_name

    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(obj = model.state_dict(), f = model_save_path)

    return model_save_path


def tensorboard_writer(experiment_name: str, model_name: str, extra: str = None, save_dir = r'.\runs') -> SummaryWriter:
    '''Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a directory constructed from the inputs; equivalent to

    SummaryWriter(log_dir = 'runs/YYYY-MM-DD/experiment_name/model_name/extra')

    :param experiment_name: Name of experiment
    :param model_name: Name of model
    :param extra: Anything extra to add to the directory; defaults to None
    :return: Instance of a writer saving to log_dir
    '''
    timestamp = datetime.now().strftime('%Y-%m-%d')

    log_dir = os.path.join(save_dir, timestamp, experiment_name, model_name)
    if extra: log_dir = os.path.join(log_dir, extra)
        
    print(f'[INFO] Created SummaryWriter, saving to: {log_dir}...')

    return SummaryWriter(log_dir = log_dir)


def download_unzip(source: str, destination: str, remove_source: bool = True) -> Path:
    '''Downloads a zipped dataset from source and unzips it at destination.

    :param source: A link to a zipped file containing data
    :param destination: A target directory to unzip data to
    :param remove_source: Whether to remove the source after downloading and extracting
    :return: pathlib.Path to downloaded data
    '''
    # Setup path to data folder
    data_path = Path('data/')
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir(): print(f'[INFO] {image_path} directory exists, skipping download.')
    else:
        print(f'[INFO] Did not find {image_path} directory, creating one...')
        image_path.mkdir(parents = True, exist_ok = True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source)
            print(f'[INFO] Downloading {target_file} from {source}...')
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            print(f'[INFO] Unzipping {target_file} data...') 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source: os.remove(data_path / target_file)
    
    return image_path



#### Info Functions ####

def summ(model: nn.Module, input_size: tuple):
    '''Shorthand for typical summary specification'''
    return summary(model = model, input_size = input_size,
                   col_names = ['input_size', 'output_size', 'num_params', 'trainable'], col_width = 20, row_settings = ['var_names'])



#### Plotting Functions ####

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions = None):
    '''Plots (matplotlib) linear training data and test data and compares predictions.
    Training data is in blue, test data in green, and predictions in red (if present).
    '''
    plt.figure(figsize = (10, 7))

    plt.scatter(train_data, train_labels, c = 'b', s = 4, label = 'Training data')
    plt.scatter(test_data, test_labels, c = 'g', s = 4, label = 'Testing data')
    if predictions is not None: plt.scatter(test_data, predictions, c = 'r', s = 4, label = 'Predictions')

    plt.legend(prop = {'size': 14})


def plot_loss_curves(train_loss: list, train_metric: list, test_loss: list, test_metric: list):
    '''Plots (matplotlib) training (and testing) curves from lists of values.
    '''
    epochs = range(len(train_loss))

    plt.figure(figsize = (15, 7))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label = 'train_loss')
    plt.plot(epochs, test_loss, label = 'test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metric, label = 'train_metric')
    plt.plot(epochs, test_metric, label = 'test_metric')
    plt.title('Performance Metric')
    plt.xlabel('Epochs')
    plt.legend()


def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    '''Plots (matplotlib) decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    '''
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to('cpu')
    X, y = X.to('cpu'), y.to('cpu')

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode(): y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1) if len(torch.unique(y)) > 2 else torch.round(torch.sigmoid(y_logits))

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap = plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


