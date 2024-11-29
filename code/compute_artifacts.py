import os,sys
from pathlib import Path
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar, cast

import time
from datetime import datetime
from collections import OrderedDict
from functools import partial
import logging 

# data or cache IO
import json
import joblib

# print, debug
from pprint import pprint

# numpy and friends
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# torch imports
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.linalg import norm as tnorm
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils


# Pytorch-lightning
import pytorch_lightning as pl
from pytorch_lightning.core  import LightningModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import Callback


# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
print("ID of assigned GPU: ", os.environ["CUDA_VISIBLE_DEVICES"])
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ', DEVICE)


# Reprlearn
from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch 
from reprlearn.utils.misc import info, now2str
from reprlearn.utils.misc import get_first_img_fp, get_img_fps, count_imgs, is_img_fp, is_valid_dir 
from reprlearn.utils.misc import read_image_as_tensor, load_pil_img


# import artifact compute functions
from reprlearn.utils.fpts import estimate_projection, compute_artifacts, compute_artifact
from reprlearn.utils.fpts import estimate_projection_fp #, compute_artifacts, compute_artifact


# Reprlearn
# -- utils
from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch
from reprlearn.utils.misc import info, now2str, get_first_img_info, count_imgs, mkdir
from reprlearn.utils.misc import set_logger, set_wandb_logger


# -- for datasets/features
from reprlearn.data.datasets.base import DatasetFromDF

from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_np, to_logscale
from reprlearn.utils.image_proc import compute_magnitude_spectrum_channelwise_np
from reprlearn.utils.embeddings import xform_for_resnet
from reprlearn.utils.embeddings import extract_feature_resnet50, extract_feature_barlowtwin
from reprlearn.utils.fpts import XFORMS
# -- models
# from reprlearn.models.densenet import DenseNet
# from reprlearn.models.googlenet import GoogleNet
# from reprlearn.models.resnet_uva import ResNet

# Helper
def show_artifact_triplet( 
    img_fp: Path, 
    proj_fp: Path,
    xform: Optional[Callable]=None,
    as_gray: Optional[bool]=False, # whether to read the fp's as gray pil image
    use_abs: Optional[bool]=False,
    save: Optional[bool]=False
    save_dir: Optional[Path]=Path('.')
) -> Tuple[plt.Figure, plt.Axes]:
    """Show  a tuple of (x_g, x_p, art), with title=label, where:
    art:= xform(x_g) - xform(x_p) 
    #label:= label of x_g (e.g., model_id or fam_id)
    
    Returns:
    -------
    f, ax
    
    """
    x_g = load_pil_img(img_fp, as_gray=as_gray)
    x_p = load_pil_img(proj_fp, as_gray=as_gray)
    if xform is not None:
        x_g = xform(x_g)
        x_p = xform(x_p)
    art = x_g - x_p

    if use_abs:
        art = art.abs()
    min_art = art.min()
    max_art = art.max()
    # show_timgs([x_g, x_p, art], 
    #            titles=['x_g', 'x_proj', f'artifact ({min_art:.2f}, {max_art:.2f})'],
    #            title=label_str, nrows=1)
    show_timg(
        timg=art, 
        title=f'artifact ({min_art:.2f}, {max_art:.2f})',
    )
    if save:
        plt.savefig(f'artrgb-triplet-{label_str}-{idx}-{now2str()}', dpi = 300)


# Globals
FEATURE_SPACE = 'art_rgb'  #art_freq #art_sl #art_ssl
# === Path to experiment dataset (GM256)
SSD_HOME = Path(os.environ.get('myssd', '~'))
TMPDIR = Path(os.environ['TMPDIR'])
JID = os.environ.get('SLURM_JOBID', -1)

# -- Data root-dir in compute node
DATA_ROOT = TMPDIR / 'gm256'                 # elif rsync is done
REAL_DATA_DIR = DATA_ROOT / 'real-celebahq256'

# === Verify
assert DATA_ROOT.exists()

# === MANIFOLD approximation
MANIFOLD_DIR = REAL_DATA_DIR
SIZE_MANIFOLD = 30_000    # max 30_000 for celeba-hq-256
MANIFOLD_FPS = get_img_fps(MANIFOLD_DIR, SIZE_MANIFOLD)
print('n_imgs on manifold: ', len(MANIFOLD_FPS))

# === MODEL_DIRS, ie. GM image dirs
D_MODEL_DIRS = {} #Dict[str,dirPath]
for model_dir in DATA_ROOT.iterdir():
    if not is_valid_dir(model_dir):
        continue
        
    model_name = model_dir.name
    D_MODEL_DIRS[model_name] = model_dir
    
D_MODEL_DIRS = dict(sorted(D_MODEL_DIRS.items()))  #dict(sorted(unsorted_dict.items()))
MODEL_NAMES = list(D_MODEL_DIRS.keys())
print('model names: ', MODEL_NAMES)

# === Define which column name specifies the class label for model attribution problem
LABEL_KEY = 'model_name'


# Set loggers
out_dir = Path('./artifacts')
mkdir(out_dir)
log_fp = out_dir / f'log-{now2str()}'
logger = set_logger(log_fp=log_fp, descr=FEATURE_SPACE)
# wb_logger = set_wandb_logger(descr=FEATURE_SPACE)



# test: compute_artifacsts
# Real images as data manifold 
ref_fps = get_img_fps(REAL_DATA_DIR, SIZE_MANIFOLD)

# compute artifact on this model's image
model_name = MODEL_NAMES[0]
img_dir = D_MODEL_DIRS[model_name]
print(f'{model_name}, {img_dir}')

x_fp = img_dir/'0000000.jpg'
x = read_image_as_tensor(img_dir/'0000000.jpg')
# x_p, _, _ = estimate_projection(x, manifold=dset)
# test est. fp
argmin_fp, x_min, d_min = estimate_projection_fp(x_fp, manifold_fps=ref_fps)
print(f'argmin_fp: {argmin_fp}, dmin: {d_min}')
f,ax = show_timg(x, title='x')
f.savefig(out_dir/f'{x_fp.name}-{now2str()}.png')

f,ax = show_timg(read_image_as_tensor(argmin_fp), title='proj_fp')
f.savefig(out_dir/ f'proj_fp-{now2str()}.png')

