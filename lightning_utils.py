'''
Collection of boilerplate and utility functions for PyTorch Lightning.
'''
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision as tv
# from torch.optim.optimizer import ParamsT # Could use instead of nn.Module as optimiser_factory argument
#     # I.e. ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder, LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler

import os
from pathlib import Path
from datetime import datetime

# import inspect
from collections import OrderedDict
from typing import Callable


class Strike(L.LightningModule):
    '''As in 'Lightning Strike', to make a PyTorch Module a LightningModule'''
    def __init__(self, model: nn.Module,
                 loss_fn: Callable[[torch.Tensor], torch.Tensor], metric_name_and_fn: tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.tensor]],
                 optimiser_factory: Callable[[nn.Module], torch.optim.Optimizer],
                 prediction_fn: Callable[[torch.Tensor], torch.Tensor],
                 learning_rate = 0.001, log_at_every_step = False):
        '''Class for turning a nn.Module into a LightningModule (a lightning strike of sorts).
        The optimiser_factory argument is a callable taking in the module, from which it extracts .parameters() and .learning_rate to produce an optimiser.
        prediction_fn is the function to be applied to model outputs to transform them into the desired prediction format, e.g. for classification logits -> probabilities -> class.

        Fields of .state_dict from by this class are the same as those of the non-wrapped .model, i.e. with no leading 'model.',
        therefore saved state_dicts can be imported into the unwrapped and wrapped model alike with .load_state_dict.
        (Additionally, 'model.' is automatically prepended if necessary when instead using Strike.load_from_checkpoint or when a Trainer resumes from a checkpoint).
        '''
        super().__init__()

        self.model = model
            # If the model form were known then its layers could be moved to this object's level rather than a nested one (not necessary but neater)
            # The procedural versions pf this are not useful since a nested nn.Sequential still exists, i.e. any of
            #   self.model = nn.Sequential(target._modules) # Preserves layer names
            #   self.model = nn.Sequential(*source.children()) # *source.modules() would return the larger container as well
        
        self.loss_fn = loss_fn
        self.metric_name, self.metric_fn = metric_name_and_fn
        self.optimiser_factory = optimiser_factory
        self.prediction_fn = prediction_fn

        self.learning_rate = learning_rate
        self.log_at_every_step = log_at_every_step
        self.train_step_outputs, self.validation_step_outputs, self.test_step_outputs = dict(), dict(), dict()

        # Allow Strike.load_from_checkpoint without restating all the parameters
        ## CANNOT WORK since multiple arguments are functions, which are not pickleable
        ## COULD SOLVE by moving them to a function producing a class without them
        # self.save_hyperparameters(ignore = ['model', 'loss_fn']) # Ignoring because already saved since nn.Modules
    
    def state_dict(self):
        # Make sure that the exported state fields look like those of the wrapped .model
        #   Interesting note: the change only seems to be required when saving a model restored from a checkpoint
        sd = super().state_dict()
        return OrderedDict({k[6:]: v for k, v in sd.items()}) if all(k[:6] == 'model.' for k in sd) else sd


    def load_state_dict(self, state_dict: os.Mapping[str, torch.Any], strict: bool = True, assign: bool = False):
        # Fields of .state_dict from by this class are the same as those of the non-wrapped .model, i.e. with no leading 'model.';
        #   therefore it only needs to be prepended when this method is called by internal methods,
        #   i.e. by Strike.load_from_checkpoint or through a Trainer resuming from a checkpoint.
        # return super().load_state_dict(state_dict if inspect.stack()[1][3] in ['_load_state', 'load_model_state_dict'] else {f'model.{k}': v for k, v in state_dict.items()}, strict, assign)
        # But can do it more generally by directly checking for the prefixes
        return super().load_state_dict(state_dict if all(k[:6] == 'model.' for k in state_dict) else OrderedDict({f'model.{k}': v for k, v in state_dict.items()}), strict, assign)

    def forward(self, x):
        return self.model(x)
    
    # No need to override these two hooks
    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #     loss.backward()
    # def optimizer_step(self, epoch, batch_idx, optimiser, optimizer_idx):
    #     optimiser.step()

    def training_step(self, batch, batch_idx):
        loss, metric, x_hat, y = self._common_step(batch, batch_idx)
        self.train_step_outputs = dict(prefix = 'train', loss = loss, metric = metric)
        return loss
    
    def on_train_epoch_end(self):
        self._common_epoch_end_step(self.train_step_outputs)
    
    def validation_step(self, batch, batch_idx):
        loss, metric, x_hat, y = self._common_step(batch, batch_idx)
        self.validation_step_outputs = dict(prefix = 'val', loss = loss, metric = metric)
        return loss
    
    def on_validation_epoch_end(self):
        self._common_epoch_end_step(self.validation_step_outputs)

    def test_step(self, batch, batch_idx):
        loss, metric, x_hat, y = self._common_step(batch, batch_idx)
        self.test_step_outputs = dict(prefix = 'test', loss = loss, metric = metric)
        return loss
    
    def on_test_epoch_end(self):
        self._common_epoch_end_step(self.test_step_outputs)

    def _common_step(self, batch, batch_idx):
        x, y = batch 
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, y)
        metric = self.metric_fn(x_hat, y)
        return loss, metric, x_hat, y
    
    def _common_epoch_end_step(self, outputs):
        self.log_dict({f'{outputs["prefix"]}_loss': outputs['loss'], f'{outputs["prefix"]}_{self.metric_name}': outputs['metric']}, prog_bar = True, on_step = self.log_at_every_step, on_epoch = True)
        outputs.clear() # Freeing memory is suggested in the docs, though it is trivial in this class

    def predict_step(self, batch, batch_idx):
        x, y = batch 
        x_hat = self.forward(x)
        preds = self.prediction_fn(x_hat)
        return preds

    def configure_optimizers(self):
        return self.optimiser_factory(self)



class LocalImageDataModule(L.LightningDataModule):
    def __init__(self, folders: str | Path | dict[str, str | Path], transform: tv.transforms.Compose,
                 batch_size: int, num_workers: int = os.cpu_count(), split: tuple[float, float, float] = (0.7, 0.2, 0.1)):
        super().__init__()
        '''Return a LightningDataModule for a local image folder (or folders) for classification purposes.
        Images are expected to be in subfolders named by their classes.
        In the str or Path folders cases, the folder content is checked for subfolders called train, test and valid (yes, in this order for consistency), and if any is present they are treated as the list input,
        however, if none is present, then the split argument is required, containing a tuple of proportions to allocate to training, validation and testing datasets.
        In the dict folders case the keys are expected to be in ['train', 'valid', 'test'].
        The class names are from the first folder and assumed to be consistent across the others.
        '''

        ########### Could relax requirement to train and test and then produce a validate dataset from the training one #########

        self.prefixes = ['train', 'valid', 'test']

        data_path = None
        if isinstance(folders, (str, Path)):
            data_path = Path(folders)
            folders = {sub: full_sub for sub in self.prefixes if (full_sub := data_path / sub).is_dir()}
        elif not isinstance(folders, dict): raise ValueError('Please provide a folders argument of types str | Path | dict[str, str | Path].')

        assert set(folders.keys()).issubset(self.prefixes), f'Exactly the {self.prefixes} folders are required; {folders.keys()} were provided.'
        if len(folders) == 3: folders = folders
        elif len(folders) == 0 and data_path is not None:
            assert sum(split) == 1
            folders = (data_path, dict(zip(self.prefixes, split)))
        else: raise ValueError(f'All of {self.prefixes} subfolders are required for the single-folder folders argument; only {folders.keys()} were provided.')

        self.folders = folders
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.classes = None

    # def prepare_data(self):
    #     '''Not currently implemented. Mostly meant for downloading data.'''
    #     pass

    def setup(self, stage):
        if isinstance(self.folders, tuple):
            all_data = tv.datasets.ImageFolder(self.folders[0], transform = self.transform)
            self.classes = all_data.classes
            self.train_ds, self.val_ds, self.test_ds = random_split(all_data, self.folders[1])
        else:
            if stage == 'fit':
                self.train_ds, self.val_ds = [tv.datasets.ImageFolder(self.folders[k], transform = self.transform) for k in self.prefixes[:-1]]
                self.classes = self.train_ds.classes
            if stage == 'test':
                self.test_ds = tv.datasets.ImageFolder(self.folders[self.prefixes[-1]], transform = self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, pin_memory = True, persistent_workers = True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, pin_memory = True, persistent_workers = True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, pin_memory = True, persistent_workers = True)



class IteratedLearningRateFinder(LearningRateFinder):
    def __init__(self, at_epochs: list[int], *args, **kwargs):
        '''CURRENTLY FAILS AT THE 2ND OCCURRENCE (despite being suggested in the docs: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateFinder.html)
        The lr finding tuns at epoch 0 regardless of whether 0 is in at_epochs.
        E.g. for periodic lr adjustments pass [e for e in range(epochs) if e % period == 0]'''
        super().__init__(*args, **kwargs)
        self.at_epochs = at_epochs

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.at_epochs or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)



class TBLogger(L.loggers.TensorBoardLogger):
    def __init__(self, experiment_name: str, model_name: str, extra: str = None, save_dir = r'.\runs', log_graph = False, default_hp_metric = True, prefix = '', **kwargs):
        '''Saves TensorBoard logs to save_dir/YYYY-MM-DD/experiment_name/model_name/extra.
        (It maps them to L.loggers.TensorBoardLogger's save_dir/name/version/sub_dir/)
        '''
        super().__init__(save_dir = save_dir, name = datetime.now().strftime('%Y-%m-%d'),
            version = experiment_name, sub_dir = os.path.join(model_name, extra),
            log_graph = log_graph, default_hp_metric = default_hp_metric, prefix = '', **kwargs)


