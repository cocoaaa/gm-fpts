#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

import os,sys
from datetime import datetime
from collections import OrderedDict
from functools import partial
sys.dont_write_bytecode = True
from IPython.core.debugger import set_trace as breakpoint


# In[ ]:

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


# In[ ]:


import torch 

# Select Visible GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Import ReprLearn and TileMani packages

# In[ ]:


import reprlearn as rl


# In[ ]:


from reprlearn.visualize.utils import get_fig, show_timg, show_timgs, show_npimgs, show_batch, make_grid_from_tensors
from reprlearn.utils.misc import info, now2str, today2str, get_next_version_path, n_iter_per_epoch
from reprlearn.utils.misc import count_imgs


# In[ ]:


# import helpers for computing spectrum of rgb images
from reprlearn.utils.image_proc import compute_magnitude_spectrum_channelwise_fp
from reprlearn.utils.image_proc import compute_avg_normed_log_mag_spectrum_channelwise
from reprlearn.utils.image_proc import compute_avg_mag_spectrum_channelwise
from reprlearn.utils.image_proc import compute_magnitude_spectrum_of_grayscale_fp, compute_magnitude_spectrum_of_grayscale_np
from reprlearn.utils.image_proc import compute_avg_magnitude_spectrum_of_grayscale
from reprlearn.utils.image_proc import grayscale_median_filter, channelwise_median_filter, normalize_to_01, normalize_to_negposone
from reprlearn.utils.misc import get_first_npimg, get_first_img_fp

# ## log to wandb

# ## Path to data root dirs
# data_root = Path('/data/datasets/CNN_synth_testset/')
data_root = Path('/data/hayley-old/Github/Fingerprints/Fourier-Discrepancies-CNN-Detection/Samples')
# import json
# json_fp = '/data/hayley-old/ModelSpace/logs/sample_dirs_2022-06-01.json' #todo: use args.json_fp
# with open(json_fp, 'r') as f:
#     dict_img_dirpaths = json.load(f) #dict(model_name, img_dirpath)

exp_id = now2str()
out_root = Path(f'../outs/C2021/{exp_id}')
print('Exp_id: ', exp_id)


def main(args):
    # Let's first count images on the models with 2deepth
    # (a sincle image category and 0_real/1_fake subdirectories)
    records = [] # each row item 
    for model_dir in data_root.iterdir():
        # print(model_name, Path(sample_dir).exists())
        img_dirname = model_dir.stem
        img_dirpath = model_dir
        
        #collect the count and image dirpath info
        n_imgs = count_imgs(img_dirpath)
        first_img = get_first_npimg(img_dirpath)
        img_shape = first_img.shape
        record = {'img_dirname': img_dirname,
                'n_imgs': n_imgs,
                'img_shape': img_shape,
                'first_img': wandb.Image(first_img),
                'img_dirpath': img_dirpath
                }
        records.append(record)

    print('Num records:', len(records))

    # create a pd.Dataframe from the records
    df = pd.DataFrame.from_records(records)

    # ## Add the column for `avg_highpass_spectrum` to the table for ForenSynths test dataset
    # Now, let's compute the average high-pass filtered spectra for each img_dir and add the average spectrum as new column to the table
    # F's for images in rgb space
    df = df.set_index('img_dirname')
    df['F_rgb'] = pd.Series() # Add an empty column
    df['low_F_rgb'] = pd.Series() # Add an empty column
    df['high_F_rgb'] = pd.Series() # Add an empty column

    # F's for images in grayscale
    df['F_gray'] = pd.Series() #avg of grayscale F's
    df['low_F_gray'] = pd.Series() #avg of blurred grayscale F's
    df['high_F_gray'] = pd.Series() #avg of blurred grayscale F's


    target_size = None
    sample_size = args.sample_size #2_000
    kernel_size = args.kernel_size #median_filter
    failed_img_dirs = []

    # create output folders for graysclae, rgb spectrums (if not existing)
    out_dir_f_rgb = out_root/'f_rgb'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_f_rgb.exists():
        out_dir_f_rgb.mkdir(parents=True)
        print('Created: ', out_dir_f_rgb)
    out_dir_low_f_rgb = out_root/'low_f_rgb'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_low_f_rgb.exists():
        out_dir_low_f_rgb.mkdir(parents=True)
        print('Created: ', out_dir_low_f_rgb)
    out_dir_high_f_rgb = out_root/'high_f_rgb'/ f'ks-{kernel_size}_nimgs-{sample_size}'
    if not out_dir_high_f_rgb.exists():
        out_dir_high_f_rgb.mkdir(parents=True)
        print('Created: ', out_dir_high_f_rgb)

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
        
        # ----------------------------------------------------------------------- #
        # spectrum in rgb space 
        # ----------------------------------------------------------------------- #
        #spectrum of origianl rgb img (no grayscale conversion, no blurring)
        f_rgb = compute_avg_mag_spectrum_channelwise(
                img_dirpath, 
                target_size=None,
                max_n_imgs=sample_size,
                transform=None
        )
        log_f_rgb = np.log(f_rgb)
        normed_log_f_rgb = normalize_to_01(log_f_rgb)

        # spectrum of blurred rgb
        low_f_rgb = compute_avg_mag_spectrum_channelwise(
                img_dirpath, 
                target_size=None,
                max_n_imgs=sample_size,
                transform=lambda x: channelwise_median_filter(x, kernel_size)
            )
        log_low_f_rgb = np.log(low_f_rgb)
        normed_log_low_f_rgb = normalize_to_01(log_low_f_rgb)

        # spectrum of highpass rgb
        high_f_rgb = compute_avg_mag_spectrum_channelwise(
                img_dirpath, 
                target_size=None,
                max_n_imgs=sample_size,
                transform=lambda x: x - channelwise_median_filter(x, kernel_size)
            )
        log_high_f_rgb = np.log(high_f_rgb)
        normed_log_high_f_rgb = normalize_to_01(log_high_f_rgb)
        
        # add to the dataframe
        df.loc[img_dirname, 'F_rgb'] = wandb.Image(normed_log_f_rgb)
        df.loc[img_dirname, 'low_F_rgb'] = wandb.Image(normed_log_low_f_rgb)
        df.loc[img_dirname, 'high_F_rgb'] = wandb.Image(normed_log_high_f_rgb)
            
        # save to file
        out_fp_f_rgb = out_dir_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_f_rgb, normed_log_f_rgb)

        out_fp_low_f_rgb = out_dir_low_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_low_f_rgb, normed_log_low_f_rgb)
        
        out_fp_high_f_rgb = out_dir_high_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_high_f_rgb, normed_log_high_f_rgb)
        print('Done: ', img_dirname)


        # ----------------------------------------------------------------------- #
        # spectrum of images in grayscale 
        # ----------------------------------------------------------------------- #
        f_gray = compute_avg_magnitude_spectrum_of_grayscale(img_dirpath,
                                                        max_n_imgs=sample_size) #grayscale
        # for spectrum of grayscale + median filtering
        low_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: grayscale_median_filter(x, kernel_size=kernel_size) 
        )
        # for spectrum of grayscale, highpass img
        high_f_gray = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=sample_size,
            transform=lambda x: x - grayscale_median_filter(x, kernel_size=kernel_size) 
        ) 
        normed_log_f_gray = normalize_to_01(np.log(f_gray))
        normed_log_low_f_gray = normalize_to_01(np.log(low_f_gray))
        normed_log_high_f_gray = normalize_to_01(np.log(high_f_gray))
        
        
        # add to dataframe to be sent to wandb
        df.loc[img_dirname, 'F_gray'] = wandb.Image(normed_log_f_gray)
        df.loc[img_dirname, 'low_F_gray'] = wandb.Image(normed_log_low_f_gray)
        df.loc[img_dirname, 'high_F_gray'] = wandb.Image(normed_log_high_f_gray)
        
        # save to file
        out_fp_f_gray = out_dir_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_f_gray, normed_log_f_gray)

        out_fp_low_f_gray = out_dir_low_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_low_f_gray, normed_log_low_f_gray)
        
        out_fp_high_f_gray = out_dir_high_f_gray/ f'{img_dirname}.png'
        plt.imsave(out_fp_high_f_gray, normed_log_high_f_gray)

        print('Done gray: ', img_dirname)
        print('Done all: ', img_dirname)
        print('-'*80)

        
    # Upload this dataframe as wandb table
    #create a wandb Table 
    wb_df_forensynth = df.drop(columns=['img_dirpath'])
    wb_df_forensynth['img_shape'] = wb_df_forensynth['img_shape'].apply(str)
    # - set index to row index (so that the img_dirname appears on wandb table in the UI)
    wb_df_forensynth = wb_df_forensynth.reset_index()

    wb_tbl  = wandb.Table(dataframe=wb_df_forensynth)
    tbl_key = f"spectrum-MyGMDataset-{exp_id}"

    # Upload to wandb 
    wandb.log({tbl_key: wb_tbl})
    print('Uploaded to wandb table key: ', tbl_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ks", "--kernel_size", required=True, type=int)
    parser.add_argument("-n", "--sample_size", required=True, type=int)

    args = parser.parse_args()
    main(args)




# run
# cd scripts
# nohup python compute_spectrum_c2021.py -ks 3 -n 2000 >>  log.c2021.txt &
# nohup python compute_spectrum_c2021.py -ks 9 -n 2000 >>  log.c2021.txt &
