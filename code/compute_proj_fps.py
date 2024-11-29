#!/usr/bin/env python
# coding: utf-8

# Import libs
import argparse
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
import logging
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



# Reprlearn
from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch 
from reprlearn.utils.misc import info, now2str
from reprlearn.utils.misc import get_first_img_fp, get_img_fps, count_imgs, is_img_fp, is_valid_dir 
from reprlearn.utils.misc import read_image_as_tensor, load_pil_img, adjust_root_dir

# import artifact compute functions
from reprlearn.utils.fpts import estimate_projection_batch

# -- for datasets/features
from reprlearn.data.datasets.base import DatasetFromDF
from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_np, to_logscale
from reprlearn.utils.image_proc import compute_magnitude_spectrum_channelwise_np
# from reprlearn.utils.embeddings import xform_for_resnet
# from reprlearn.utils.embeddings import extract_feature_resnet50, extract_feature_barlowtwin



def main(
    server: str, #arya or turing
    feature_space: str, # 'rgb' or 'freq'
    size_manifold: int, # number of datapoints used to find project of each img in gm256
    mode: str, #'train', 'val', 'test
    split_id: int, #df split id
    start_i: int, # start index of the subset df to process
    split_size: Optional[int]=1000,
    end_i: Optional[int]=None,    # df_{mode}.iloc[start_i:end_i] will be processed
    batch_size: Optional[int]=8, # for dataloader of real-dataset (manifold/reference data)
    num_workers: Optional[int]=1, # for dataloader of real-dataset (manifold/reference data)
    log_fp: Optional[str]=None,
    log_every: Optional[int]=10,  # log to the file, how many `img_fp` has found its `proj_fp`
    save_every: Optional[int]=50, # save df so far with `proj_fp` every this many `img_fp` processed 
    use_cpu: Optional[int]=False, # if True, use CPU even if cuda/GPU is available   
):
    """ df_{mode}.iloc[start_i:end_i] will be processed
    # Specify which df to process 
    # mode = 'test'

    # Speifiy subset of df_{mode} to process
    # split_id = 13
    start_i = 12_000 # todo: check how many already done
    split_size = 1000
    end_i = start_i + split_size
    """
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    print("ID of assigned GPU: ", os.environ.get("CUDA_VISIBLE_DEVICES", -1))
    # os.environ["CUDA_VISIBLE_DEVICES"]="2"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_cpu:
        DEVICE = "cpu"
    print('DEVICE: ', DEVICE)
    
    server = server.lower()
    if end_i is None:
        end_i = start_i + split_size
        
    print(f'pid: {os.getpid()}')

    # Set logger
    # Log filepath
    log_fp = log_fp or f'./log-art-rgb-{mode}-split{split_id}-{now2str()}.txt'
    print('log_fp: ', log_fp)
    # log_fp = f'./log-art-rgb-valset-split{split_id}-{now2str()}.txt'
    # log_fp = "log-art-fft-20230426-150532.txt"
    # print('log_fp: ', log_fp)
    logging.basicConfig(filename=log_fp,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logging.info(f"Compute artifacts on {feature_space} space for df_{mode}!")

    logger = logging.getLogger(f'art-{feature_space}-{mode}set')

    # to stream to both file and stderr (console)
    logging.getLogger().addHandler(logging.StreamHandler())

    # disable matplotlib debug errors to be logged
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    logging.info(f'pid: {os.getpid()}')


    # GLOBALS:
    # -- Move data to cluster's compute node
    # srcdir = SSD_HOME / 'Datasets/gm256_from_arya'
    # dstdir = TMPDIR 
    # !rsync -aznP {str(srcdir)} {str(dstdir)}

    # === Path to experiment dataset (GM256)
    # -- Data root-dir in compute node
    if server == 'arya':
        JID='0'
        DATA_ROOT = Path('/data/datasets/neurips23_gm256/')   # if on arya
        
    elif server == 'turing':
        JID = os.environ['SLURM_JOBID']
        SSD_HOME = Path(os.environ.get('myssd', '~'))
        TMPDIR = Path(os.environ['TMPDIR'])
        # DATA_ROOT = SSD_HOME / 'Datasets/gm256_from_arya'    # if rsync is not yet done
        DATA_ROOT = TMPDIR / 'gm256_from_arya'                 # elif rsync is done
    else:
        raise ValueError(f'server must be either arya or turing: {server}')

    REAL_DATA_DIR = DATA_ROOT / 'real-celebahq256'

    # === Verify
    assert DATA_ROOT.exists()
    # print('Subdirs: ')
    # !ls {DATA_ROOT}

    # print('\nNum. real datapts: ')
    # !ls {REAL_DATA_DIR} | wc -l


    # === MANIFOLD approximation
    MANIFOLD_DIR = REAL_DATA_DIR
    # size_manifold = 30_000    # max 30_000 for celeba-hq-256
    MANIFOLD_FPS = get_img_fps(MANIFOLD_DIR, size_manifold)
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


    # ## Run: find proj_fp of each `img_fp` in df (e.g., df_train, df_val, df_test)
    # Given `estimate_projection_fp` is working correctly:

    # ### 1. Load train/val/test datasets from cached csv files
    # where precomputed files are saved
    CACHE_DIR = Path('./cache')
    # === Load precomputed data-splits (as pandas.DataFrame) ===
    # Define Dataset, Dataloaders
    # load csv's as pd.DataFrames
    # columns: ['img_fp', 'fam_name', 'model_name']
    
    df_all_fp = CACHE_DIR / 'data_splits' / 'df-all-gm256.csv'
    df_train_fp = CACHE_DIR / 'data_splits' / 'df-train-gm256.csv'
    df_val_fp = CACHE_DIR / 'data_splits' / 'df-val-gm256.csv'
    df_test_fp = CACHE_DIR / 'data_splits' / 'df-test-gm256.csv'


    df_all = pd.read_csv(df_all_fp, index_col=0)
    df_train = pd.read_csv(df_train_fp, index_col=0)
    df_val = pd.read_csv(df_val_fp, index_col=0)
    df_test = pd.read_csv(df_test_fp, index_col=0)

    dfs = {
        'all': df_all,
        'train': df_train,
        'val': df_val,
        'test': df_test
    }

    if server != 'arya':
        # adjust rootdir if server is not arya
        # - [ ] apply it here (see the workflow in 14-00)
        for _m, _df in dfs.items():
            dfs[_m] = adjust_root_dir(
                df=_df,
                path_cols=['img_fp'], # ['img_fp, 'proj_fp'],
                new_root_dir=DATA_ROOT,
                index_to_keep=-2
            )
            # dfs[mode] = df  # weird. without this assingment dfs[mode] doesn't get updated... 
            

            
        # reassign to declared variables
        df_all = dfs['all']
        df_train = dfs['train']
        df_val = dfs['val']
        df_test = dfs['test']
        
    print('len(df_all), len(df_train), len(df_val), len(df_test)')
    print(len(df_all), len(df_train), len(df_val), len(df_test))

    

    # column name in train/val/test DataFrames to be used as label of each image
    LABEL_KEY = 'model_name' 
    N_CLASSES = len(df_train[LABEL_KEY].unique())
    print('N classes: ', N_CLASSES)


    # ## 2. Define data transforms (to each feature space: RGB, Freq, SL, SSL)
    # Define transforms for each feature space
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
    #     'sl': xform_sl,
    #     'ssl': xform_ssl,
    }


    # ### [HERE!]  New: use this version
    # === Create a dataloader of the all real-dataset (30k)
    # size_manifold = 30_000
    df_all_real = df_all.loc[df_all.fam_name=='real'].iloc[:size_manifold]
    logger.info(f'df_all_real: {len(df_all_real)}')

    # --- Create a Dataset 
    # for the real-celebahq256 in original rgb 
    real_dset_rgb = DatasetFromDF(
        df=df_all_real,
        label_key=LABEL_KEY,
        xform=xform_rgb
    )

    # --- Dataloader for dataset of "Reals"
    # batch_size = 16
    shuffle = False
    # num_workers = 4
    pin_memory = True

    real_dl_rgb = DataLoader(
        dataset=real_dset_rgb,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    def process(df: pd.DataFrame,
                device: Union[torch.device, str],
                mode: str, # train/val/test
                log_every: int=50,
                save_every: int=100

            ):
        dset_name = 'gm256'
        feature_space = 'rgb'  #'fft'
        
        start = time.time()
        
        proj_fps = [ ] 
        start_index = df.index[0]
        end_index = df.index[-1]
        n_total = len(df)
        logger.info(f"RUN: Processing img_fps in df: {start_index} to {end_index}...")

        for i, x_g_fp in enumerate(df['img_fp']):

            fam_name = df['fam_name'].iloc[i]
            # no need to process reals:
            if fam_name == 'real':
                min_fp = x_g_fp
            else:
                x_g = xform_rgb(load_pil_img(x_g_fp))       ## todo: xform_rgb vs. xform_freq
                result = estimate_projection_batch(x_g, 
                                                dl_manifold=real_dl_rgb, ## todo: real_dl_rgb vs real_dl_freq
                                                device=device)
                min_fp = result['min_fp']
                # min_dist = result['min_dist']
            proj_fps.append(min_fp)
            
            if (i+1)%log_every == 0:
                logger.info(f'{i+1}th image: {(i+1)*100/n_total:.2f} % ...')
            if (i+1)%save_every == 0:
                # save so far
                temp = df.iloc[:i+1].copy()
                temp['proj_fp'] = proj_fps
                temp.to_csv(f'df-{mode}-{start_index}-{start_index+i}-with-proj-{feature_space}-{dset_name}-onek-{now2str()}.csv')
                logger.info(f'Saved so far processed df: with proj_fp for df_{mode}: [{start_index},{start_index+i}] ')

            

        # done:
        df['proj_fp'] = proj_fps
        end = time.time()
        logger.info(f'Done: computing proj_fp for images in df_{mode}!')
        logger.info(f'Took: {(end-start) / 60.} mins; size manifold={size_manifold}; n_img_fps={n_total}')
        
        # last save:
        df.to_csv(f'df-{mode}-{start_index}-{end_index}-with-proj-{feature_space}-{dset_name}-onek-{now2str()}.csv')
        logger.info(f'Last: Saved all processed df: with proj_fp for df_{mode}!')
        logger.info(f'=== Done: art-{feature_space} for  df_{mode}: {start_index}:{end_index} ===')


    # yes: functionalized; 
    # no: splitting of df to smaller chunks and parallelizing
    # --- Process all img_fps
    # TODO: replace the function to be apply with the get_proj_fp below
    # proj_result = estimate_projection_batch(x_g, dl_manifold=real_dl_rgb)

    ## !! -- Specify which df to process ---!!
    ## Get df split
    df = dfs[mode].iloc[start_i:end_i].copy() #todo: for debug/speed testing
    
    # debug
    # print('debug: ', df.img_fp.iloc[0])
    # print('debug: ', dfs[mode].img_fp.iloc[0])
    # print('mode: ', mode)
    # breakpoint()
    
    process(df, 
            device=DEVICE,
            mode=mode,
            log_every=log_every,
            save_every=save_every,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cpu", action="store_true", help="Use cpu, ignore GPU even if available")
    parser.add_argument("--server", required=True, type=str,
                        help="arya or turing")
    parser.add_argument("--feature_space", required=True, type=str,
                        help="artifact will be computed on this feature space: rgb or freq")
    parser.add_argument("--size_manifold", required=False, type=int, default=30_000,
                        help="Number of datapoints in real dataset to be used as manifold. For celeba-hq-256, use 30_000 to use all data")     
    
    # specify which df to process
    parser.add_argument("--mode", required=True, type=str,
                        help="which df_{mode} to process: must be one of train, val, test")
    parser.add_argument("--split_id", required=True, type=str,
                        help="assign id for this df-split for logging")
    parser.add_argument("--start_i", required=True, type=int,
                        help="index of the first row of the subset of df to process")
    parser.add_argument("--end_i", required=False, type=int, default=None,
                        help="index of the last row -1 of the subset of df to process")
    parser.add_argument("--split_size", required=True, type=int,
                        help="size of the subset of df we are processing; will be ignored if end_i is given; else end_i = start_i + split_size")                        

    # manifold batch loading 
    parser.add_argument("--batch_size", required=False, type=int, default=4,
                        help="batch_size for the dataloader of real-dataset as manifold/reference data for searching closest point to a query image")                        
    parser.add_argument("--num_workers", required=False, type=int, default=1,
                        help="num_workers for the dataloader of real-dataset as manifold/reference data for searching closest point to a query image")                        

    # logging
    parser.add_argument("--log_fp", required=False, type=str, default=None,
                        help="filepath to the log txt")                        
    parser.add_argument("--log_every", required=False, type=int, default=30,
                        help="log to the file, how many `img_fp` has found its `proj_fp`")
    parser.add_argument("--save_every", required=False, type=int, default=50,
                        help="save df so far with `proj_fp` every this many `img_fp` processed ")


    # Parse cli arguments
    args = parser.parse_args()
    
    print('args: ')
    pprint(args)

    main(
        server=args.server,
        feature_space=args.feature_space,
        size_manifold=args.size_manifold,
        mode=args.mode,
        split_id=args.split_id,
        start_i=args.start_i,
        split_size=args.split_size,
        end_i=args.end_i,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_fp=args.log_fp,
        log_every=args.log_every,
        save_every=args.save_every,
        use_cpu=args.use_cpu,
    ) 

"""
# How to use:
# == Pre-req:
# Make sure /data/xxxxx/data_splits has these csv's:
# 'df-all-gm256.csv', 'df-train-gm256.csv', 'df-val-gm256.csv', 'df-test-gm256.csv'

# == Run these commands on shell:
export mode=      # train or val or test
export split_id=  # todo: this is only for logging purpose; match with the table i wrote on paper
export start_i=   # todo: important - df[start_i: start_i + split_size] will be processed

# optional
export split_size=1000 # default
export batch_size=16  # default
export num_workers=4  # default
export log_fp=XXX     # eg.'Logs/log-test-rgb-df-...'


python compute_proj_fps.py --server arya --feature_space rgb \
    --mode $mode --split_id $split_id --start_i $start_i \
    # add optional args here if want to change \
    &

"""