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


import wandb
wandb.login()


data_dir_wang2020 = Path('/data/datasets/CNN_synth_testset/')
exp_id = now2str()
out_root = Path(f'../outs/{exp_id}')
print('Exp_id: ', exp_id)

# Process Forensynths testset from Wang2020
models_3deep = ['stylegan', 'cyclegan', 'progan', 'stylegan2']


def main(args):
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
            
    # now, process the image-dirs that are 3 level deep from the root
    for model_dir in data_dir_wang2020.iterdir():
        if not model_dir.is_dir() or model_dir.stem not in models_3deep:
            print(model_dir, ": skipping...")
            continue
        model_name = model_dir.stem
            
        for model_cat_dir in model_dir.iterdir():
            if not model_cat_dir.is_dir():
                continue
            cat_name = model_cat_dir.stem
            
            # 0/1 class labels: this is the image dir
            for label_dir in model_cat_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                
                label_name = label_dir.stem #img_dir
                dset_name = f'{model_name}-{cat_name}-{label_name}' #unique name for this model-class: e.g., starget-1_fake, cyclegan_apple
                
                #collect the count and image dirpath info
                n_imgs = count_imgs(label_dir)
                first_img = get_first_npimg(label_dir)
                img_shape = first_img.shape
                record = {'img_dirname': dset_name,
                        'n_imgs': n_imgs,
                        'img_shape': img_shape,
                        'first_img': wandb.Image(first_img),
                        'img_dirpath': label_dir
                        }
                records.append(record)
                


    print('num records:', len(records))

    # create a pd.Dataframe from the records
    df_forensynth = pd.DataFrame.from_records(records)

    # ## Add the column for `avg_highpass_spectrum` to the table for ForenSynths test dataset
    # Now, let's compute the average high-pass filtered spectra for each img_dir and add the average spectrum as new column to the table
    df_forensynth = df_forensynth.set_index('img_dirname')
    df_forensynth['F_rgb'] = pd.Series() # Add an empty column
    df_forensynth['low_F_rgb'] = pd.Series() # Add an empty column
    df_forensynth['high_F_rgb'] = pd.Series() # Add an empty column

    # Wang2020, with median_blurring (2022-07-11 (mon))

    target_size = None
    sample_size = args.sample_size #2_000
    kernel_size = args.kernel_size #median_filter
    failed_img_dirs = []
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

    for row in df_forensynth.itertuples():
        img_dirpath = row.img_dirpath
        img_dirname = row.Index

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
        df_forensynth.loc[img_dirname, 'F_rgb'] = wandb.Image(normed_log_f_rgb)
        df_forensynth.loc[img_dirname, 'low_F_rgb'] = wandb.Image(normed_log_low_f_rgb)
        df_forensynth.loc[img_dirname, 'high_F_rgb'] = wandb.Image(normed_log_high_f_rgb)
            
        # save to file
        out_fp_f_rgb = out_dir_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_f_rgb, normed_log_f_rgb)

        out_fp_low_f_rgb = out_dir_low_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_low_f_rgb, normed_log_low_f_rgb)
        
        out_fp_high_f_rgb = out_dir_high_f_rgb/ f'{img_dirname}.png'
        plt.imsave(out_fp_high_f_rgb, normed_log_high_f_rgb)
        print('Done: ', img_dirname)
        # break
        # show for debug
    #     f,ax = plt.subplots(1,2)
    #     ax[0].imshow(log_avg_low_f_rgb)
    #     ax[0].set_title('log(avg lowpass spectrum')
    #     ax[1].imshow(normed_log_avg_low_f_rgb)
    #     ax[1].set_title('normed to 01 log(avg lowpass spectrum)')
    #     break



    # ### Compute the first image's spectrum as original in rgb, and after blurring the rgb
    # F's for grayscale
    df_forensynth['F_gray'] = pd.Series() #avg of grayscale F's
    df_forensynth['low_F_gray'] = pd.Series() #avg of blurred grayscale F's
    df_forensynth['high_F_gray'] = pd.Series() #avg of blurred grayscale F's

    # # F's for rgb
    # df_forensynth['F_rgb'] = pd.Series()
    # df_forensynth['low_F_rgb'] = pd.Series()

    # ### Compute avg spectrum as original in grayscale, and after blurring the grayscale
    # compute the first image's spectrum as original in grayscale, and after blurring the grayscale
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

    for row in df_forensynth.itertuples():
        img_dirpath = row.img_dirpath
        img_dirname = row.Index
        
        # for spectrum of grayscale img
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

    #     info(img)
    #     info(blurred_img)
    #     breakpoint()
        # show for debug
    #     fig, ax = plt.subplots(1,2, figsize=(10,5))
    #     fig.suptitle(img_dirname)
        
    #     ax[0].imshow(normalize_to_01(np.log(f_img)))
    #     ax[0].set_title('log(f_rgb)')
        
    #     ax[1].imshow(normalize_to_01(np.log(low_f)))
    #     ax[1].set_title('log(low_f)')
            
    #     fig.show()
    #     fig.savefig(out_dir/f'{img_dirname}-{now2str()}.png')
        
        # add to dataframe to be sent to wandb
        df_forensynth.loc[img_dirname, 'F_gray'] = wandb.Image(normed_log_f_gray)
        df_forensynth.loc[img_dirname, 'low_F_gray'] = wandb.Image(normed_log_low_f_gray)
        df_forensynth.loc[img_dirname, 'high_F_gray'] = wandb.Image(normed_log_high_f_gray)
        

        print('Done: ', img_dirname)
        
    #     break


    # Upload this dataframe as wandb table
    #create a wandb Table 
    wb_df_forensynth = df_forensynth.drop(columns=['img_dirpath'])
    wb_df_forensynth['img_shape'] = wb_df_forensynth['img_shape'].apply(str)
    # - set index to row index
    wb_df_forensynth = wb_df_forensynth.reset_index()
    # wb_df_forensynth  = wb_df_forensynth.set_index(pd.RangeIndex(start=0, stop=len(wb_df_forensynth)))

    wb_tbl  = wandb.Table(dataframe=wb_df_forensynth)
    tbl_key = f"spectrum-Forensynths-testset-{exp_id}"

    # Upload to wandb 
    wandb.log({tbl_key: wb_tbl})
    print('Uploaded to wandb table key: ', tbl_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ks", "--kernel_size", required=True, type=int)
    parser.add_argument("-n", "--sample_size", required=True, type=int)

    args = parser.parse_args()
    main(args)



