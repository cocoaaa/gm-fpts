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
from IPython.core.debugger import set_trace

sys.dont_write_bytecode = True



# numpy and friends
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# torch imports
import torch 
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader



# Reprlearn
from reprlearn.utils.misc import info, now2str, get_first_img_info, count_imgs
# visalize in 2dim
from reprlearn.visualize.utils import reduce_dim_and_plot


def compute_and_plot_feature_space(
    encoder: nn.Module, # input -> feature
    dl: DataLoader,
    device,
    data_split: str, # train or test; for save name
    model_name: str, # for save name
    save_precomputed: bool=False, # to pickle the computed features and labels with joblib.dump
    save_tsne_plot: bool=True,
    legend: Optional='full',
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    encoder.to(device)
    
    dset = dl.dataset
    n_samples = len(dset)
    n_iters = int(np.ceil(n_samples/dl.batch_size))
    print(f'n_samples, n_iters: {n_samples}, {n_iters}')

    features = [] 
    labels = [] 
    start = time.time()
    #todo: print_every
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dl):
            if i >= n_iters:
                break
                
            batch_z = encoder(batch_x.to(device))
    #         print(batch_z.shape) #(bs, dim_z=2048, 1,1)
            batch_z.squeeze_()
    #         print(batch_z.shape) #(bs, dim_z=2048)
            features.append(batch_z.detach().cpu())
            
            
    #         print(batch_y.shape) #(bs,)
            labels.append(batch_y.cpu())
            
            
    features = torch.vstack(features) # (n_samples, dim_z=2048)
    labels = torch.concat(labels)     # (n_samples,)

    end = time.time()
    print(f'Done encoding inputs to learned features. It took: {(end-start)/60.0} mins')
        
    
    # Save features, labels for later 
    if save_precomputed:
        joblib.dump(
            features, 
            f'./features_for_{data_split}_onek_dset_{model_name}-{now2str()}.pkl'
        )
        joblib.dump(
            labels, 
            f'./labels_for_{data_split}_onek_dset_{model_name}-{now2str()}.pkl'
        )
    print('feature shape: ', features.shape)
    print('labels shape: ', labels.shape)
    
    labels_str = [ dset.i2c[label.item()] for label in labels ]
    print('labels eg: ', labels_str[:10])
    
     # 1. tsne
    feature_type = model_name
    dim_reduction = 'tsne'
    config_tsne = {
        'n_components':2,
        'perplexity': 30.0,
        'n_iter': 1000,
        'metric': 'euclidean'
    }

    # tsne plot
    figsize = (10,10)
    f_tsne, ax_tsne = plt.subplots(figsize=figsize)

    t0 = time.time()
    reduce_dim_and_plot(
        data=features,
        labels=labels_str,
        algo_name=dim_reduction,
        ax=ax_tsne,
        title=f'Feature space: learned-{feature_type}-{data_split}, {dim_reduction}',
        legend=legend,
        **config_tsne
    )
    print("Done tsne.")
    print(f'Took: {(time.time() - t0)/60. } mins')
    
    if save_tsne_plot:
        fp = f"tsne-{feature_type}-trained-on-onek_dset-{data_split}-{now2str()}.png"
        plt.savefig(fp)
        print(f'Saved tsne plot of the learned embedding of datapts in {data_split} set!') 
    return features, labels

    
    
def freeze(net:nn.Module):
    """inplace op to freeze each and all parameters of the net
    """
    for p in net.parameters():
        p.requires_grad = False
