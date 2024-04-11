'''Torchvision and related utility functions'''

import torch
import torchvision as tv
from torch.utils.data import DataLoader
import timm # Here just to be exported

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import base64
import altair as alt
import matplotlib.pyplot as plt # REMOVE IN FAVOUR OF ALTAIR

import os
import io
from pathlib import Path
from PIL import Image
# from itertools import batched # in Python>=3.12


def image_dataloaders(folders: str | Path | list[str | Path], transform: tv.transforms.Compose, batch_size: int, num_workers: int = os.cpu_count()) -> tuple[list[DataLoader], list[str]]:
    '''Return PyTorch DataLoaders and class names for the given folder or list of folders (with expected subfolders named by class).
    In the non-list folders case, the folder content is checked for subfolders called train, test and valid (yes, in this order for consistency), and if any is present they are treated as the list input.
    The first folder is assumed to be the training data and will therefore produce a shuffling dataloader, while the others will not.
    The class names are from the first folder and assumed to be consistent across the others.
    '''
    if isinstance(folders, (str, Path)):
        data_path = Path(folders)
        folders = subfolders if (subfolders := [full_sub for sub in ['train', 'valid', 'test'] if (full_sub := data_path / sub).is_dir()]) else [folders]

    datasets = [tv.datasets.ImageFolder(folder, transform = transform) for folder in folders]
    dataloaders = [DataLoader(ds, batch_size = batch_size, shuffle = i == 0, num_workers = num_workers, pin_memory = True, persistent_workers = True) for i, ds in enumerate(datasets)]

    return dataloaders, datasets[0].classes


def plot_img_preds(model: torch.nn.Module, image_path: str, class_names: list[str], transform: tv.transforms, device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    '''Plot one image with its prediction and probability as the title.
    '''
    img = Image.open(image_path)

    model.to(device)
    model.eval()
    with torch.inference_mode(): pred_logit = model(transform(img).unsqueeze(dim = 0).to(device)) # Prepend "batch" dimension (-> [batch_size, color_channels, height, width])
    pred_prob = torch.softmax(pred_logit, dim = 1)
    pred_label = torch.argmax(pred_prob, dim = 1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_label]} | Prob: {pred_prob.max():.3f}")
    plt.axis(False)

    # Change text colour based on correctness?


def record_image_preds(image_paths: str | list[str], model: torch.nn.Module, transform: tv.transforms.Compose, class_names: list[str],
                       sort_by_correctness = True, device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    '''Generate a dataframe of paths, true classes, (single) predicted classes and their confidence.
    Column names: path, true_class, pred_class, pred_prob, correct.
    If sort_by_correctness, then the dataframe is sorted by increasing correctness and confidence, i.e. first by prediction correctness and then by its probability,
    with wrong predictions first, and both wrong and right by decreasing confidence.
    If a single string is given as image_paths, then all */*.jpg and */*.png matches from it are used instead.
    '''
    true_classes, pred_classes, pred_probs, correctness, image_data = [], [], [], [], []

    if isinstance(image_paths, str): image_paths = list(Path(image_paths).glob('*/*.jpg')) + list(Path(image_paths).glob('*/*.png'))

    for path in tqdm(image_paths):
        img = Image.open(path)

        model.eval()
        with torch.inference_mode(): pred_logit = model(transform(img).unsqueeze(0).to(device)) # Prepend "batch" dimension (-> [batch_size, color_channels, height, width])
        pred_prob = torch.softmax(pred_logit, dim = 1)
        pred_label = torch.argmax(pred_prob, dim = 1)

        true_classes.append(class_name := path.parent.stem)
        pred_classes.append(pred_class := class_names[pred_label.cpu()])
        pred_probs.append(pred_prob.unsqueeze(0).max().cpu().item())
        correctness.append(class_name == pred_class)


    res = pd.DataFrame(dict(path = [str(p) for p in image_paths], true_class = true_classes, pred_class = pred_classes, pred_prob = pred_probs, correct = correctness))
    return res.sort_values(by = ['correct', 'pred_prob'], ascending = [True, False]) if sort_by_correctness else res


def base64_image_formatter(image_or_path: Image.Image | str) -> str:
    '''Generate a base64-encoded string representation of the given image (or path).
    Example usecase: a dataframe meant for Altair contains PIL images (or their paths) in a column, in which case pass this temporary dataframe to the alt.Chart:
        `df.assign(image = df.image_or_path_column.apply(base64_image_formatter))`
    '''
    if isinstance(image_or_path, str): image_or_path = Image.open(image_or_path)
    with io.BytesIO() as buffer: # Docs: https://altair-viz.github.io/user_guide/marks/image.html#use-local-images-as-image-marks
        image_or_path.save(buffer, format = 'PNG')
        data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{data}'


def image_pred_grid(image_df: pd.DataFrame, ncols = 4, img_width = 200, img_height = 200, allow_1_col_reduction = True):
    '''Create an Altair plot displaying a grid of images and their predicted classes, highlighting incorrect predictions.
    image_df is expected to have the columns: path, true_class, pred_class, pred_prob, correct.
    If allow_1_col_reduction and the last row (by the given ncols) is at least half empty and using ncols-1 would not increase rows, then ncols-1 is used instead.
    '''
    # Docs: https://altair-viz.github.io/user_guide/compound_charts.html
    # Opened issue on making it easier through alt.Facet: https://github.com/altair-viz/altair/issues/3398

    ncols = min(ncols, len(image_df))
    nrows = 1 + len(image_df) // ncols
    # If the last row is at least half empty and could reduce columns without increasing rows, do so
    if allow_1_col_reduction and nrows > 1 and len(image_df) % ncols <= ncols / 2 and 1 + len(image_df) // (ncols - 1) == nrows: ncols -= 1

    expanded_df = image_df.assign(
        image = image_df.path.apply(base64_image_formatter),
        title = image_df.pred_class + ' - ' + image_df.pred_prob.map(lambda p: f'{p:.2f}'),
        index = image_df.index
    )

    base = alt.Chart(expanded_df).mark_image(width = img_width, height = img_height).encode(url = 'image:N')
    chart = alt.vconcat()
    for row_indices in (expanded_df.index[i:i + ncols] for i in range(0, len(expanded_df), ncols)): # itertools.batched(expanded_df.index, ncols) in Python>=3.12
        row_chart = alt.hconcat()
        for index in row_indices:
            row_chart |= base.transform_filter(alt.datum.index == index).properties(
                title = alt.Title(expanded_df.title[index], fontSize = 17, color = 'green' if expanded_df.correct[index] else 'red'))
        chart &= row_chart

    ## Version with no subplots (but no titles)
    # chart = alt.Chart(image_df.assign( # vv cannot trust the df index since it might not be ordered
    #     row = np.arange(len(image_df)) // ncols, col = np.arange(len(image_df)) % ncols # Could use the transform_compose block for this, but no // in the alt.expr language
    # )).mark_image(width = img_width, height = img_height).encode(
    #     alt.X('col:O', title = None, axis = None), alt.Y('row:O', title = None, axis = None), url = 'image:N'
    # ).properties(
    #     width = img_width * 1.1 * ncols, height = img_height * 1.1 * nrows
    # )

    ## Version with faceting (but not coloured titles (no titles in fact, but non-coloured headers))
    # chart = alt.Chart(image_df.assign(
    #     image = image_df.path.apply(base64_image_formatter),
    #     title = image_df.pred_class + ' - ' + image_df.pred_prob.map(lambda p: f'{p:.2f}')
    # )).mark_image(width = img_width, height = img_height).encode(url = 'image:N'
    # ).facet( # Header fields: https://altair-viz.github.io/user_guide/generated/core/altair.Header.html
    #     alt.Facet('title:N', header = alt.Header(labelFontSize = 17, labelColor = 'red')).title('Prediction and Confidence'), columns = ncols, title = 'Hi'
    # )

    return chart





# import torchvision
# import matplotlib.pyplot as plt
# # Plot the top 5 most wrong images
# for row in top_5_most_wrong.iterrows():
#   row = row[1]
#   image_path = row[0]
#   true_label = row[1]
#   pred_prob = row[2]
#   pred_class = row[3]
#   # Plot the image and various details
#   img = torchvision.io.read_image(str(image_path)) # get image as tensor
#   plt.figure()
#   plt.imshow(img.permute(1, 2, 0)) # matplotlib likes images in [height, width, color_channels]
#   plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}")
#   plt.axis(False);



