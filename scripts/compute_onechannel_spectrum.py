#!/usr/bin/env python
# coding: utf-8

# ## Load libraries
import os,sys
from datetime import datetime
from collections import OrderedDict
from functools import partial
sys.dont_write_bytecode = True
from IPython.core.debugger import set_trace as breakpoint


import argparse
import json
import pandas as pd
import joblib
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
import matplotlib.pyplot as plt
import skimage as ski
from skimage import io
from PIL import Image

from pathlib import Path
from typing import Any,List, Set, Dict, Tuple, Optional, Iterable, Mapping, Union, Callable, TypeVar

from pprint import pprint
from tqdm import tqdm

import torch 


from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch, make_grid_from_tensors
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
from reprlearn.utils.misc import count_imgs


# import helpers for computing spectrum of rgb images
from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_fp, compute_magnitude_spectrum_of_grayscale_np
from reprlearn.utils.image_proc import grayscale_median_filter, channelwise_median_filter, normalize_to_01, normalize_to_negposone, clip_floatimg
from reprlearn.utils.image_proc import crop_center
from reprlearn.utils.misc import get_first_npimg, get_first_img_fp


def _main_test(args):
    target_size = None
    sample_size = args.sample_size #2_000
    kernel_size = args.kernel_size #median_filter
    channel_ind = args.channel_ind # channel (0,1, or 2) to use to show the spectrums

    # Let's first count images on the models with 2deepth
    # (a sincle image category and 0_real/1_fake subdirectories)
    records = [] # each row item 
    for model_dir in data_dir_wang2020.iterdir():
        if not model_dir.is_dir() or model_dir.stem in models_3deep:
            print(model_dir, ": skipping...")
            continue
        model_name = model_dir.stem
            
        for model_class_dir in model_dir.iterdir():
            if not model_class_dir.is_dir():
                continue
            model_class_name = model_class_dir.stem
            
            #collect the count and image dirpath info
            dset_name = f'{model_name}-{model_class_name}' #unique name for this model-class: e.g., starget-1_fake, cyclegan_apple
            n_imgs = count_imgs(model_class_dir)
            first_img = get_first_npimg(model_class_dir)
            img_shape = first_img.shape
            record = {'img_dirname': dset_name,
                    'n_imgs': n_imgs,
                    'img_shape': img_shape,
                    'first_img': wandb.Image(first_img),
                    'img_dirpath': model_class_dir
                    }
            records.append(record)


    failed_img_dirs = []
    out_dir_f_gray = out_root/'f_gray'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_f_gray.exists():
        out_dir_f_gray.mkdir(parents=True)
        print('Created: ', out_dir_f_gray)

    for row in df.itertuples():
        img_dirpath = row.img_dirpath
        img_dirname = row.Index
        #spectrum of origianl rgb img (no grayscale conversion, no blurring)
    
        # -------------------------------------------------------------------------------------------
        # Spectrums of images in grayscale
        # -------------------------------------------------------------------------------------------
        # for spectrum of grayscale img
        f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=None
            ) 
        # for spectrum of grayscale + median filtering
        low_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: grayscale_median_filter(x, kernel_size=kernel_size) 
        )
        # for spectrum of grayscale, highpass img
        # with clipping
        high_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: clip_floatimg(
                x - grayscale_median_filter(x, kernel_size=kernel_size) 
            )
        ) 

    

def _main(args):
    sample_size = args.sample_size #2_000
    kernel_size = args.kernel_size #median_filter

    # ## Add the column for `avg_highpass_spectrum` to the table for ForenSynths test dataset
    # Now, let's compute the average high-pass filtered spectra for each img_dir and add the average spectrum as new column to the table
    
    # Output folders for spectrum of images in grayscale 
    out_dir_f_gray = out_root/'f_gray'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_f_gray.exists():
        out_dir_f_gray.mkdir(parents=True)
        print('Created: ', out_dir_f_gray)
    out_dir_low_f_gray = out_root/'low_f_gray'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_low_f_gray.exists():
        out_dir_low_f_gray.mkdir(parents=True)
        print('Created: ', out_dir_low_f_gray)
    out_dir_high_f_gray = out_root/'high_f_gray'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_high_f_gray.exists():
        out_dir_high_f_gray.mkdir(parents=True)
        print('Created: ', out_dir_high_f_gray)

    for row in df.itertuples():
        img_dirpath = row.img_dirpath
        img_dirname = row.Index

        # -------------------------------------------------------------------------------------------
        # Spectrums of images in grayscale
        # -------------------------------------------------------------------------------------------
        # for spectrum of grayscale img
        f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=None
            ) 
        # for spectrum of grayscale + median filtering
        low_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: grayscale_median_filter(x, kernel_size=kernel_size) 
        )
        # for spectrum of grayscale, highpass img
        # with clipping
        high_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: clip_floatimg(
                x - grayscale_median_filter(x, kernel_size=kernel_size) 
            )
        ) 
        out_fp_f_gray = out_dir_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_f_gray, normed_log_f_gray)

        out_fp_low_f_gray = out_dir_low_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_low_f_gray, normed_log_low_f_gray)
        
        out_fp_high_f_gray = out_dir_high_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_high_f_gray, normed_log_high_f_gray)

        print('Done gray: ', img_dirname)

        print('Done all: ', img_dirname)
        print('-'*80)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel_ind", required=True, type=int)
    parser.add_argument("-ks", "--kernel_size", required=True, type=int)
    parser.add_argument("-n", "--sample_size", required=True, type=int)

    args = parser.parse_args()
    # main_test(args)

    main(args)

# run
# cd scripts
# nohup python compute_onechannel_spectrum_forensynths_testset.py -c 0 -ks 3 -n 100  >> log.txt &


