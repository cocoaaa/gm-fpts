#!/usr/bin/env python
# coding: utf-8

# ## Import libs
import os,sys
from pathlib import Path
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar, cast

import time
from datetime import datetime
from collections import OrderedDict
from functools import partial

# data or cache IO
import json
import joblib

# print, debug
from pprint import pprint
from IPython.display import display, HTML # for pretty-print pandas dataframe
from IPython.core.debugger import set_trace as breakpoint

sys.dont_write_bytecode = True


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
from reprlearn.utils.misc import read_image_as_tensor

# -- import artifact compute functions
from reprlearn.utils.fpts import estimate_projection, compute_artifacts, compute_artifact
from reprlearn.utils.fpts import estimate_projection_fp #, compute_artifacts, compute_artifact

# -- utils
from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch
from reprlearn.utils.misc import info, now2str, get_first_img_info, count_imgs
from reprlearn.utils.misc import load_pil_img, adjust_root_dir
# -- for datasets/features
from reprlearn.data.datasets.base import DatasetFromDF
from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_np, to_logscale
from reprlearn.utils.image_proc import compute_magnitude_spectrum_channelwise_np
# from reprlearn.utils.embeddings import xform_for_resnet
# from reprlearn.utils.embeddings import extract_feature_resnet50, extract_feature_barlowtwin

# -- models
# from reprlearn.models.densenet import DenseNet
# from reprlearn.models.googlenet import GoogleNet
from reprlearn.models.resnet_uva import ResNet


# Global setting
# representation space for buidling data manifold
feature_space = 'rgb' #'fft'


import wandb
wandb.login()

from pytorch_lightning.loggers import WandbLogger

wb_logger = WandbLogger(
    entity='usc-isi-vimal', # use to log the metrics to a specific wandb team
    project='Fingerprinting-GMs',
    # project='model-space',    # organize/group runs into a parent project
    log_model='all',          # 'all': log all new checkpoints during training
    name=f'exp1-art_{feature_space}-with-reals-{now2str()}', #name of this run of script
    tags=['exp1', 'gm-fpts', f'art_{feature_space}']
) 




# Set logger to file in disk and std out
import logging
log_fp = f'./exp1-art-{feature_space}-{now2str()}.txt'
print('log_fp: ', log_fp)

logging.basicConfig(filename=log_fp,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# logging.info("Running PCA on artifacts on {feature_space} space!")
# logger = logging.getLogger('PCA-art_{feature_space}')


logging.info("Train and eval gm-classifier using artifacts on {feature_space} space!")
logger = logging.getLogger('Exp1-art_{feature_space}')


# GLOBALS:
# === Path to experiment dataset (GM256)
SSD_HOME = Path(os.environ.get('myssd', '~'))
TMPDIR = Path(os.environ['TMPDIR'])
JID = os.environ.get('SLURM_JOBID', -1)
# -- Move data to cluster's compute node
# srcdir = SSD_HOME / 'Datasets/gm256_from_arya'
# dstdir = TMPDIR 
# !rsync -aznP {str(srcdir)} {str(dstdir)}

# -- Data root-dir in compute node
# DATA_ROOT = SSD_HOME / 'Datasets/gm256_from_arya'    # if rsync is not yet done
DATA_ROOT = TMPDIR / 'gm256_from_arya'                 # elif rsync is done
REAL_DATA_DIR = DATA_ROOT / 'real-celebahq256'

# === Verify
assert DATA_ROOT.exists()
# print('Subdirs: ')
# !ls {DATA_ROOT}

# print('\nNum. real datapts: ')
# !ls {REAL_DATA_DIR} | wc -l


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


LABEL_KEY = 'model_name'



# ## 2. Define data transforms (to each feature space: RGB, Freq, SL, SSL)

# -- for datasets/features
from reprlearn.data.datasets.base import DatasetFromDF
from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_np
from reprlearn.utils.image_proc import compute_magnitude_spectrum_channelwise_np, to_logscale
# from reprlearn.utils.embeddings import xform_for_resnet
# from reprlearn.utils.embeddings import extract_feature_resnet50, extract_feature_barlowtwin

# == RGB
xform_rgb = transforms.ToTensor()


# == Freq
# fft-gray
xform_freq_gray = transforms.Compose([
    # pil_rgb -> pil_gray -> np float_gray
    lambda pil_rgb: np.array(pil_rgb.convert('L')) / 255.0,
    
    # np float_gray -> fft (h,w)
    compute_magnitude_spectrum_of_grayscale_np,  #all-pass
#         compute_magnitude_spectrum_channelwise_np,  #all-pass (h,w,3)

    # to logsclae (??)
    to_logscale,
    
    # if grayscale fft,
    # -- 2dim nparray (fft) -> 3dim tensor (1, h,w)  
    lambda fft_np: torch.tensor(fft_np[None, :, :],
                               dtype=torch.float),
    # (1, h, w) to (3, h,w) by repeating the channles
    # try: with or without this channel-replication
    # 20230420-200447: trying without.
#     lambda fft_onechannel: torch.vstack(
#         [fft_onechannel, fft_onechannel, fft_onechannel]
#     )
])

# fft-rgb
xform_freq_rgb = transforms.Compose([
    # pil_rgb -> np float
    lambda pil_rgb: np.array(pil_rgb) / 255.0,
    
    # np float -> fft-channelwise (h,w,3)
    compute_magnitude_spectrum_channelwise_np,  #all-pass (h,w,3)

    # to logsclae (??)
    to_logscale,
    
    # For channelwise fft, swap the axis: (h,w,nc) --> (nc, h, w)
    lambda fft_np: torch.tensor(fft_np.transpose((2,0,1)),
                                dtype=torch.float),
])

# !! -- todo: specify which one -- !!
# xform_freq = xform_freq_gray
xform_freq = xform_freq_rgb


# == RGB + Freq (either fft-gray or fft-rgb)
xform_rgb_freq = transforms.Compose([
    
    #pil --> concat channel-dim (rgb, fft_gray or fft_rgb)
    lambda im: torch.vstack([
        xform_rgb(im), 
        xform_freq(im)
    ])
])


# # == SL
# xform_sl = transforms.Compose([
#     xform_for_resnet,
#     lambda t: extract_feature_resnet50(input=t)
# ])

# # == SSL
# xform_ssl = transforms.Compose([
#     xform_for_resnet,
#     lambda t: extract_feature_barlowtwin(input=t)
# ])



XFORMS = {
    'rgb': xform_rgb,
    'freq': xform_freq,
    'rgb_freq': xform_rgb_freq,
    # 'sl': xform_sl,
    # 'ssl': xform_ssl,
}

# IN_C = 1
# IN_C = 4 #rgb + freq_gray 
# IN_C = 3 #rgb or freq_rgb
# IN_C = 6 # rgb + fft-channelwise
IN_CS = {
    'gray': 1,
    'rgb': 3,
    'freq_gray': 1,
    'freq_rgb': 3,
    'freq': 3,
    'rgb+freq': 6,
}
IN_C = IN_CS[feature_space]


# ## 3. Load df_train/val/test_with_proj_rgb
def is_valid_csv_fp(fp: Path):
    return fp.is_file() and fp.suffix =='.csv'

train_with_proj_csv_dir = Path('./Outputs/df_splits_with_proj_rgb_final/train')
# val_with_proj_csv_dir = Path('./Outputs/df_splits_with_proj_rgb_final/val')
test_with_proj_csv_dir = Path('./Outputs/df_splits_with_proj_rgb_final/test')

# === Load df_train_with_proj =======
# load each split csv and concat to a single df_train_with_proj
df_train_with_proj = None
fps = sorted([fp for fp in train_with_proj_csv_dir.iterdir()
              if is_valid_csv_fp(fp)])
for fp in fps:
    df_split = pd.read_csv(fp, index_col=0)
    
    if df_train_with_proj is None:
        df_train_with_proj = df_split.copy()
    else:
        df_train_with_proj = pd.concat([df_train_with_proj, df_split])

print(f"len(df_train_with_proj): {len(df_train_with_proj)}")

# === Load df_test_with_proj =======
# load each split csv and concat to a single df_train_with_proj
df_test_with_proj = None
fps = sorted([fp for fp in test_with_proj_csv_dir.iterdir()
              if is_valid_csv_fp(fp)])
for fp in fps:
    df_split = pd.read_csv(fp, index_col=0)
    
    if df_test_with_proj is None:
        df_test_with_proj = df_split.copy()
    else:
        df_test_with_proj = pd.concat([df_test_with_proj, df_split])
    
print(f"len(df_test_with_proj): {len(df_test_with_proj)}")


from reprlearn.data.datasets.utils import stratified_split_by_ratio
split_ratio =  3.0/7.0
# e.g., if n_val: n_train = 3: 7, split_raio : = n_val/n_train = 3/7 

split_seed = 123
split_shuffle = False
ratio_train_per_label=0.7

df_train_with_proj, df_val_with_proj = stratified_split_by_ratio(
    df_all = df_train_with_proj, 
    label_key=LABEL_KEY,
    ratio_train_per_label=ratio_train_per_label,
    shuffle=split_shuffle,
    seed=split_seed,
)


# === check if split created proper df_train and df_val
# df_train
print('=== df_train_with_proj')
print(f'len: {len(df_train_with_proj)}')
# count by each GM 
display(df_train_with_proj.groupby('model_name').count())
# count by each fam
display(df_train_with_proj.groupby('fam_name').count())


# == df_val
print('=== df_val_with_proj')
print(f'len: {len(df_val_with_proj)}')
# count by each GM 
display(df_val_with_proj.groupby('model_name').count())
# count by each fam
display(df_val_with_proj.groupby('fam_name').count())


# preprocess:
# apply the adjust_root_dir before passing in the `df_train_with_proj

# ===  df_train_with_proj
print('=== Before: ')
print(df_train_with_proj[['img_fp', 'proj_fp']].head(3))

df_train_with_proj = adjust_root_dir(
    df=df_train_with_proj,
    path_cols=['img_fp', 'proj_fp'],
    new_root_dir=DATA_ROOT,
    index_to_keep=-2
)
    
print('=== After: ')
print(df_train_with_proj[['img_fp', 'proj_fp']].head(3))
# print(df_train_with_proj[['img_fp', 'proj_fp']].iloc[[0,-1]])


# === df_val_with_proj
print('=== Before: ')
print(df_val_with_proj[['img_fp', 'proj_fp']].head(3))

df_val_with_proj = adjust_root_dir(
    df=df_val_with_proj,
    path_cols=['img_fp', 'proj_fp'],
    new_root_dir=DATA_ROOT,
    index_to_keep=-2
)
    
print('=== After: ')
print(df_val_with_proj[['img_fp', 'proj_fp']].head(3))
# print(df_val_with_proj[['img_fp', 'proj_fp']].iloc[[0,-1]])



# === df_test_with_proj
print('=== Before: ')
print(df_test_with_proj[['img_fp', 'proj_fp']].head(3))

df_test_with_proj = adjust_root_dir(
    df=df_test_with_proj,
    path_cols=['img_fp', 'proj_fp'],
    new_root_dir=DATA_ROOT,
    index_to_keep=-2
)
    
print('=== After: ')
print(df_test_with_proj[['img_fp', 'proj_fp']].head(3))
# print(df_test_with_proj[['img_fp', 'proj_fp']].iloc[[0,-1]])






# Check: simple data statistics on df
dfs = {
    'train': df_train_with_proj,
    'val': df_val_with_proj,
    'test': df_test_with_proj
}
for _mode, _df in dfs.items():
    print(f'=== {_mode}')
    # count by each GM 
    display(_df.groupby('model_name').count())

    # count by each fam
    display(_df.groupby('fam_name').count())


# column name in train/val/test DataFrames to be used as label of each image
LABEL_KEY = 'model_name' 
N_CLASSES = len(df_train_with_proj[LABEL_KEY].unique())
print('N classes: ', N_CLASSES)

class ArtDatasetFromDF(DatasetFromDF):
    def __init__(self,
                 df: pd.DataFrame,
                 label_key: str,  # which column to be used as target_labels(y)
                 col_img_fp: str, # name of column that has img filepaths in gm256
                 col_proj_fp: str, # name of column that has filpaths to projection of each img_fp to real dataset
                 xform: Optional[Callable]=None,
                 target_xform: Optional[Callable]=None,
                 n_channels: Optional[int]=3,
                ):
        """assumes img (data being read from the filepaths) are 3-dim of either 
        (3, h, w) or (1, h, w)
        - n_channels = 3 (default) or 1
        
        """
        super().__init__(
            df=df,
            label_key=label_key,
            xform=xform,
            target_xform=target_xform
        )
        # sets: 
        # self.df, self.label_key, 
        # self.label_set, self.c2i, self.i2c,
        # self.xform, self.target_xform 
        self.as_gray = (n_channels == 1)
        self.col_img_fp = col_img_fp
        self.col_proj_fp = col_proj_fp
        
        assert len(df[col_img_fp]) == len(df[col_proj_fp])
        
    def __len__(self) -> int:
        return len(self.df[self.col_img_fp])
    
    def __getitem__(self, idx: int) -> Tuple:
        """returns  a tuple of (art, label) where
        art  : xform(x_g) - xform(x_p) 
        label: label of x_g (e.g., model_id or fam_id)
        """
        img_fp = self.df.iloc[idx][self.col_img_fp]
        proj_fp = self.df.iloc[idx][self.col_proj_fp]
        x_g = load_pil_img(img_fp, as_gray=self.as_gray)
        x_p = load_pil_img(proj_fp, as_gray=self.as_gray)
        if self.xform is not None:
            x_g = self.xform(x_g)
            x_p = self.xform(x_p)
        art = x_g - x_p
        
        label_str = self.df.iloc[idx][self.label_key]
        label = self.c2i[label_str]
        if self.target_xform is not None:
            label = self.target_xform(label)
        
        return art, label
    
    def show_triplet(self, 
                     idx:int, 
                     use_abs:Optional[bool]=False,
                     save:Optional[bool]=False,
                     save_dir: Optional[Path]=None,
                    ) -> Tuple[plt.Figure, plt.Axes]:
        """Show  a tuple of (x_g, x_p, art), with title=label, where:
        art  : xform(x_g) - xform(x_p) 
        label: label of x_g (e.g., model_id or fam_id)
        """
        img_fp = self.df.iloc[idx][self.col_img_fp]
        proj_fp = self.df.iloc[idx][self.col_proj_fp]
        x_g = load_pil_img(img_fp, as_gray=self.as_gray)
        x_p = load_pil_img(proj_fp, as_gray=self.as_gray)
        if self.xform is not None:
            x_g = self.xform(x_g)
            x_p = self.xform(x_p)
        art = x_g - x_p
        
        label_str = self.df.iloc[idx][self.label_key]
        label = self.c2i[label_str]
        if self.target_xform is not None:
            label = self.target_xform(label)
        
        if use_abs:
            art = art.abs()
        min_art = art.min()
        max_art = art.max()
        show_timgs([x_g, x_p, art], 
                   titles=['x_g', 'x_proj', f'artifact ({min_art:.2f}, {max_art:.2f})'],
                   title=label_str, nrows=1)
        # show_timg(
        #     timg=art, 
        #     title=f'artifact ({min_art:.2f}, {max_art:.2f})',
        # )
        if save:
            save_dir = save_dir or Path('.')
            plt.savefig(save_dir/ f'art-triplet-{label_str}-{idx}-{now2str()}', dpi = 300)

                 

# option1: datasets  with real
train_dset = ArtDatasetFromDF(
    df=df_train_with_proj,
    label_key=LABEL_KEY,
    col_img_fp='img_fp',
    col_proj_fp='proj_fp',
    xform=xform_rgb
)


# test: train dataset
print(f'len(train_dset): {len(train_dset)}')

# test: visualize some artifact_rgb datapoints in training set
# x, y = train_dset_artrgb[0]
# show_timg(x.abs(), title=y)


# test: show triplet of (x_g, x_proj, artifact) for ith img_fp
# save=True
# for i in range(11, len(train_dset), 300):
#     train_dset.show_triplet(i, use_abs=True, save=save)
# plt.close('all')


# Create val dataset:
# 1. read df splits for validatiaon set and concate them to a single dataframe
# # 2. create the artifact dataset object
val_dset = ArtDatasetFromDF(
    df=df_val_with_proj,
    label_key=LABEL_KEY,
    col_img_fp='img_fp',
    col_proj_fp='proj_fp',
    xform=xform_rgb
)

# Create test dataset:
# 1. read df splits for validatiaon set and concate them to a single dataframe
# # 2. create the artifact dataset object
test_dset = ArtDatasetFromDF(
    df=df_test_with_proj,
    label_key=LABEL_KEY,
    col_img_fp='img_fp',
    col_proj_fp='proj_fp',
    xform=xform_rgb
)
# test: test dataset
# print(f'len(test_dset): {len(test_dset)}')

# test: visualize some artifact_rgb datapoints in training set
# x, y = test_dset[0]
# show_timg(x.abs(), title=y)


# test: show triplet of (x_g, x_proj, artifact) for ith img_fp
# save=True
# # for i in range(0, len(test_dset_artrgb), 300):
# for i in [0, 150, 1150, 1450, 2400, 3400, 4400, 5400, 6400, 7400, 8400, 9400, 10400, 11400]:
#     test_dset.show_triplet(i, use_abs=True, save=save)
# plt.close('all')


# dataloaders
batch_size = 36
shuffle = True
num_workers = 4
pin_memory = True

train_dl = DataLoader(
    dataset=train_dset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory
)
    

val_dl = DataLoader(
    dataset=val_dset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)
    

test_dl = DataLoader(
    dataset=test_dset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)
plt.close('all')



# ### 2. train a classifier on art_rgb 
# ## Train a classifier 
# ### Define LightningModule
pl.seed_everything(42)

# Callbacks 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# In[49]:
MODEL_DICT = {
    'googlenet': GoogleNet,
    'resnet': ResNet,
    'densenet': DenseNet
}

def create_model(model_name, model_hparams):
    model_name = model_name.lower()
    if model_name in MODEL_DICT:
        return MODEL_DICT[model_name](**model_hparams)
    else:
        raise ValueError(
            f"Unknown model name \"{model_name}\". Available models are: {str(MODEL_DICT.keys())}"
        )
        
        


# Similarly, to use the activation function as another hyperparameter in our model, we define a "name to function" dict below:
act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


class CIFARModule(pl.LightningModule):

    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        scores = self.model(imgs)   # cocoaaa: modified to log val-loss'es
        loss = self.loss_module(scores, labels)
        pred_labels = scores.argmax(dim=-1)
        acc = (labels == pred_labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log('test_acc', acc)

# Train or load pretrained (if ckptfile names CHECKPOINT_PATH / {model_name}.ckpt exists)

# for fft as input
# feature_space = '
network_type = 'resnet' # or 'googlenet' or 'densenet'

CHECKPOINT_PATH = Path(f'./cache/ckpts-resnet-art_rgb-with-reals-20230607-195501/')
# CHECKPOINT_PATH = Path(f'./cache/ckpts-{network_type}-art_{feature_space}-with-reals-{now2str()}')
logger.info(f'Checkpoint path: {CHECKPOINT_PATH}')

def train_model(model_name, save_name=None, max_epochs=180,
                **kwargs) -> Tuple[nn.Module, dict[str, float]]:
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name
    log_dir = CHECKPOINT_PATH / save_name 
    
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=log_dir,   # Where to save models
        logger=wb_logger,
        max_epochs=max_epochs,   # How many epochs to train for if no patience is set
        enable_model_summary = False,
        enable_progress_bar = False, #         progress_bar_refresh_rate=0 # 0 to disable pbar

        callbacks=[
            ModelCheckpoint(dirpath=log_dir,
                            save_weights_only=True, 
                            save_top_k=3, 
                            monitor="val_acc",
                            # monitor='train_acc', # no val
                           mode="max"), # val_loss
        LearningRateMonitor("epoch")],  # Log learning rate every epoch
        )       
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = log_dir / 'best.ckpt'
    if pretrained_filename.is_file():
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = CIFARModule.load_from_checkpoint(str(pretrained_filename)) # Automatically loads the model with the saved hyperparameters
    else:
        print(f"No ckpt model found, training from scratch; logging to {log_dir}...")
        raise KeyError()
        pl.seed_everything(42) # To be reproducable
        model = CIFARModule(model_name=model_name, **kwargs)
        # trainer.fit(model, train_dl) #no validation
        trainer.fit(model, train_dl, val_dl) #train and validation
        model = CIFARModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
        
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_dl, verbose=False)
    test_result = trainer.test(model, dataloaders=test_dl, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print('result: ', result)
    
    return model, result
    # return model  # when no val, test datasets


resnet_model, resnet_results = train_model(model_name="resnet", 
                                           model_hparams={
                                               "in_channels": IN_C,
                                               "num_classes": N_CLASSES,
                                              "c_hidden": [16,32,64],
                                              "num_blocks": [3,3,3],
                                              "act_fn_name": "relu"}, 
                                           optimizer_name="SGD",
                                           optimizer_hparams={"lr": 0.1,
                                                              "momentum": 0.9,
                                                              "weight_decay": 1e-4})




TRAINED_MODELS = {
#     'googlenet': googlenet_model,
    'resnet': resnet_model,
    # 'densenet': densenet_model
}


# ## Conclusion and Comparison
# # After discussing each model separately, and training all of them, we can finally compare them. First, let's organize the results of all models in a table:
import tabulate
from IPython.display import display, HTML
all_models = [
#     ("GoogleNet", googlenet_results, googlenet_model),
    ("ResNet", resnet_results, resnet_model),
#     ("ResNetPreAct", resnetpreact_results, resnetpreact_model),
    # ("DenseNet", densenet_results, densenet_model)
]
table = [[model_name,
          f"{100.0*model_results['val']:4.2f}%",
          f"{100.0*model_results['test']:4.2f}%",
          "{:,}".format(sum([np.prod(p.shape) for p in model.parameters()]))]
         for model_name, model_results, model in all_models]
display(HTML(tabulate.tabulate(table, 
                               tablefmt='html',
                               headers=["Model",
#                                         "Val Accuracy",
                                        "Test Accuracy",
                                        "Num Parameters"
                                       ])))


# # Visualize feature spaces of learned classifier
# - View each model as (encoder + classification layer)
# - Map each datapt in the train/test datasets to learned feature space (embedding)
# - Show the embeddings using tSNE
# - Compute the clustering index 
from reprlearn.models.utils import get_head

# pretrained_bt = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
# # get the head of resnet50 trained in barlow-twin paper
# modules = list(pretrained_bt.children())[:-1]
# bt_head = nn.Sequential(*modules)  # same as in resent50, outputs 2048-long vector

# for p in bt_head.parameters():
#     p.requires_grad = False

# def get_head(net:nn.Module, freeze:bool=True) -> nn.Module:
#     modules = list(net.children())[:-1] # input_net and inception_blocks
#     output_net_no_fc = nn.Sequential(
#         *list(net.output_net.children())[:-1] #drop the last Linear layer
#     )
#     modules.append(output_net_no_fc)
#     head = nn.Sequential(*modules)

#     if freeze:
#         for p in head.parameters():
#             p.requires_grad = False
#     return head

def freeze(net:nn.Module):
    """inplace op to freeze each and all parameters of the net
    """
    for p in net.parameters():
        p.requires_grad = False


freeze(resnet_model.model)

resnet_head = get_head(
    net=resnet_model.model.eval(),
)


ENCODERS = {
#     'googlenet': googlenet_head,
    # 'densenet': densenet_head,
    'resnet': resnet_head
}


#test:
batch_x, bath_y = next(iter(train_dl))

# batch_z = googlenet_model.model(batch_x)
# batch_z = ENCODERS['googlenet'](batch_x) # (bs, 128)

# batch_z = densenet_model.model(batch_x)
# batch_z = ENCODERS['densenet'](batch_x) # (bs, 184)

# batch_z = resnet_model.model(batch_x)
batch_z = resnet_head(batch_x) # (bs, 64)

info(batch_z)



## Eval:
# # Evaluation: (1) visualize feature space; (2) plot confusion matrix
#  visualize learned features in tsne coords:
from helpers import compute_and_plot_feature_space

# now get features/embeddings of each image in the dataset
dls = {
    "train": train_dl,
    "val": val_dl,
    "test": test_dl
}


# !! -- todo: set which model's feature space here-- !!
# model_name = 'googlenet'
# model_name = 'densenet' 
model_name = 'resnet'

encoder = ENCODERS[model_name]

# --- main ---
# visualize feture space of each (train/val/test) dataset
# ------------
for data_split, dl in dls.items():

    compute_and_plot_feature_space(
        encoder=encoder,
        dl=dl,
        device=DEVICE,
        data_split=data_split,
        model_name=model_name,
        save_precomputed=False,
        save_tsne_plot=True,
        legend=False, # for wacv24 submission
    )

from reprlearn.evaluate.metrics import compute_confusion_matrix
from reprlearn.visualize.utils import plot_confusion_matrix
model = TRAINED_MODELS[model_name]
# model.to(DEVICE)
for data_split, dl in dls.items():
    mat = compute_confusion_matrix(
        model=model, 
        data_loader=dl, 
        device=DEVICE
    )
    plot_confusion_matrix(mat, class_names=dl.dataset.label_set)
    plt.title(f'{model_name} on {data_split} dset')
    plt.show()

    fp = f"conf-{model_name}-onek_dset-{data_split}-{now2str()}.png"
    plt.savefig(fp)


wandb.finish()




# #### Visualize the art-rgb dataset 
# ### 1. Initial viz. (tsne) of art_rgb dataset (not learned feature space
from reprlearn.visualize.utils import reduce_dim_and_plot

# === Collect all the artifact images and their labels into a numpy array
# Takes some time
dl = train_dl
n_samples = len(dl.dataset)
n_iters = int(np.ceil(n_samples/dl.batch_size))

start=time.time()
features = [] # (N, 3, h, w)
labels = []   # (N,)
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(dl):
        if i >= n_iters:
            break
        features.append(batch_x)
        labels.append(batch_y)
        
features = torch.vstack(features) # (n_samples, c,h,w)
labels = torch.concat(labels)     # (n_samples,)
end=time.time()

logger.info(f"Done: Features,labels collection took {(end-start)/60.0} mins")
logger.info("=== features")
info(features)

logger.info("=== labels")
info(labels)



## === To collect features, labels EXCEPT real's
## Dataset without reals (no_real)
# label of real = 
label_real_str = 'real-celebahq256'
label_real = train_dset.c2i[label_real_str]
logger.info(f'real label, label_index: {label_real_str}, {label_real}')

# # Takes some time
# # ...
# #
feature_space = 'artrgb'

dl = DataLoader(
    dataset=train_dset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)
    
n_samples = len(dl.dataset)
n_iters = int(np.ceil(n_samples/dl.batch_size))

start=time.time()
features_no_real = [] # (N, 3, h, w)
labels_no_real = []   # (N,)
n_real = 0
with torch.no_grad():
    for i, (batch_x, batch_y) in enumerate(dl):
        if i >= n_iters:
            break
        if batch_y[0] == label_real:
            n_real += 1
            # breakpoint()
            continue
        features_no_real.append(batch_x)
        labels_no_real.append(batch_y)
end = time.time()
logger.info(f"Done: Features,labels (no real) collection took {(end-start)/60.0} mins")
assert n_real == sum(df_train_with_proj[LABEL_KEY] == label_real_str)
        
features_no_real = torch.vstack(features_no_real) # (n_samples, c,h,w)
labels_no_real = torch.concat(labels_no_real)     # (n_samples,)

print("=== features_no_real")
info(features_no_real)

print("=== labels_no_real")
info(labels_no_real)

# ## for later: to load precomputed features, labels
# include_reals = True
# descr = 'with_reals' if include_reals else 'no_reals'

# # === Load precomputed features, labels
# if include_reals:
#     #  Option 1: features, labels (including real datapts)
#     features_out_fp = "artrgb_features_for_train_dset.pt"
#     labels_out_fp = "artrgb_labels_for_traind_dset.pt"
# else:
#     # === Option2: features, labels without real datapts
#     features_out_fp = f'./{feature_space}_features-no-real_for_train_dset_{now2str()}.pt'
#     labels_out_fp = f'./{feature_space}_labels-no-real_for_traind_dset_{now2str()}.pt'

# features = torch.load(features_out_fp)
# labels = torch.load(labels_out_fp)

features = features.numpy()
labels = labels.numpy()

features_no_real = features_no_real.numpy()
labels_no_real = labels_no_real.numpy()


# 1. Run tsne
dim_reduction = 'tsne'
config_tsne = {
    'n_components':2,
    'perplexity': 100.0,
    'n_iter': 2000,
    'metric': 'euclidean'
}

# === tsne2d-plot features with reals
figsize = (10,10)
f_tsne, ax_tsne = plt.subplots(figsize=figsize)

t0 = time.time()
reduce_dim_and_plot(
    data=features,
    labels=labels,
    algo_name=dim_reduction,
    ax=ax_tsne,
    title=f'Feature space: {feature_space}, {dim_reduction}',
    **config_tsne
)
logger.info("tsne2d plot - Done.")
logger.info('tsne2d on features with reals -- Took: ', (time.time() - t0))

ax_tsne.get_figure().savefig(
    f"viz-gm256-{feature_space}-with-reals-{dim_reduction}-{now2str()}.png",
    dpi=300
)

# === tsne2d-plot features w/o reals
figsize = (10,10)
f_tsne, ax_tsne = plt.subplots(figsize=figsize)

t0 = time.time()
reduce_dim_and_plot(
    data=features_no_real,
    labels=labels_no_real,
    algo_name=dim_reduction,
    ax=ax_tsne,
    title=f'Feature space (w/o reals): {feature_space}, {dim_reduction}',
    **config_tsne
)
logger.info("tsne2d plot (no reals) - Done.")
logger.info(f'tsne2d on features w/o reals -- Took: {(time.time() - t0)}')

ax_tsne.get_figure().savefig(
    f"viz-gm256-{feature_space}-no-reals-{dim_reduction}-{now2str()}.png",
    dpi=300
)
# ==== ISOMAP
# (2.a) isomap on features with real data
# dim-reduction using isomap
# parameters
dim_reduction = 'isomap'
config_isomap = {
    "n_neighbors":30
}

# plot: dataset inclduing real labelled datapts
f_iso, ax_iso = plt.subplots(figsize=figsize)

t0 = time.time()
reduce_dim_and_plot(
    data=features,
    labels=labels,
    algo_name=dim_reduction,
    ax=ax_iso,
    title=f'Feature space: {feature_space}, {dim_reduction}',
    **config_isomap
)
logger.info("isomap - done.")
logger.info('Took: ', (time.time() - t0))

# save figure
ax_iso.get_figure().savefig(
    f"viz-gm256-{feature_space}-with-reals-{dim_reduction}-{now2str()}.png",
    dpi=300,
)



# (2.b) isomap on features without real data
# dim-reduction using isomap
# parameters
dim_reduction = 'isomap'
config_isomap = {
    "n_neighbors":30
}

# plot
f_iso, ax_iso = plt.subplots(figsize=figsize)

t0 = time.time()
reduce_dim_and_plot(
    data=features_no_real,
    labels=labels_no_real,
    algo_name=dim_reduction,
    ax=ax_iso,
    title=f'Feature space (no reals): {feature_space}, {dim_reduction}',
    **config_isomap
)
logger.info("isomap (no reals) - Done.")
logger.info('isomap (no reals) - Took: ', (time.time() - t0))

# save figure
ax_iso.get_figure().savefig(
    f"viz-gm256-{feature_space}-no-reals-{dim_reduction}-{now2str()}.png",
    dpi=300,
)




# 3.a LLE on features with real datapts
# dim-reduction using LLE
dim_reduction = 'lle'
config_lle = {
    "n_neighbors":30
}

# plot
f_lle, ax_lle = plt.subplots(figsize=figsize)
t0 = time.time()
reduce_dim_and_plot(
    data=features,
    labels=labels,
    algo_name=dim_reduction,
    ax=ax_lle,
    title=f'Feature space: {feature_space}, {dim_reduction}',
    **config_lle
)
logger.info("lle (with reals) - Done.")
logger.info('lle (with reals) - Took: ', (time.time() - t0)) 
# save figure
ax_lle.get_figure().savefig(
    f"viz-gm256-{feature_space}-with-reals-{dim_reduction}-{now2str()}.png",
    dpi=300,  #600 1200
)



# 3.b LLE on features without real datapts
# dim-reduction using LLE
dim_reduction = 'lle'
config_lle = {
    "n_neighbors":30
}

# plot
f_lle, ax_lle = plt.subplots(figsize=figsize)
t0 = time.time()
reduce_dim_and_plot(
    data=features_no_real,
    labels=labels_no_real,
    algo_name=dim_reduction,
    ax=ax_lle,
    title=f'Feature space: {feature_space}, {dim_reduction}',
    **config_lle
)
    
# save figure
ax_lle.get_figure().savefig(
    f"viz-gm256-{feature_space}-no-reals-{dim_reduction}-{now2str()}.png",
    dpi=300,  #600 1200
)
logger.info("lle (no reals) - Done.")
logger.info('lle (no reals) - Took: ', (time.time() - t0))



#  run PCA, get K components, visualize the K componentes
from sklearn import decomposition
from reprlearn.visualize.utils import plot_2d_coords
dim_reduction = 'pca'
n_components=10
pca = decomposition.PCA(
    n_components=n_components, svd_solver="randomized", whiten=True
)

# === features,labels including real datapts
t0 = time.time()

reduced_features = pca.fit_transform(
    features.reshape(len(features),-1)
)
plot_2d_coords(
    data_2d=reduced_features,
    labels=labels
)
logger.info("pca (with reals) - Done.")
logger.info('pca (with reals) - Took: ', (time.time() - t0))

# show the K components
# pca_estimator.components_[:n_components]
comps = pca.components_[:n_components].reshape(n_components, 3, 256,256)
comps = comps.transpose((0,-2,-1,1))
print(f'PCA components: {comps.shape}')

f_pca, ax_pca = show_npimgs(comps,
            title=f'Principle components of {feature_space}-with-reals')
ax_pca.get_figure().savefig(
    f"viz-gm256-{feature_space}-with-reals-{dim_reduction}-{now2str()}.png",
    dpi=300,  #600 1200
)