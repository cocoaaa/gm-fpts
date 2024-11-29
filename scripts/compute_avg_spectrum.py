import argparse
from typing import Union, Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import joblib
from cytoolz import valmap
from functools import partial
import matplotlib as mpl
# mpl.use('Agg') # http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
import matplotlib.pyplot as plt

from reprlearn.utils.image_proc import compute_avg_magnitude_spectrum_of_grayscale, clip_floatimg, grayscale_median_filter
from reprlearn.visualize.utils import show_npimgs, show_dict_with_colorbar
from reprlearn.utils.misc import mkdir, now2str

from IPython.core.debugger import set_trace as breakpoint
from pprint import pprint

# Globals
FILTER_TYPES = ['allpass', 'highpass', 'lowpass']
FFT_NORM_TYPES = ['backward', 'ortho', 'forward']
       
RUN_ID = now2str('')

       
# plot helper
def get_positive_min_max(d: Dict) -> Tuple[float, float]:
    vals = np.array(list(d.values()))
    vals = np.ma.masked_less_equal(vals, 0, copy=False)
    
#     print('d: ', d)
#     print('vals: ', vals)
    return vals.min(), vals.max()
                    
def plot_dict_spectra_logscale(dict_to_show: Dict[str, np.ndarray], 
                               save: bool,
                               out_fp: Union[Path,str],
                               cmap=None, 
                               **kwargs
                               ):

    vmin, vmax = get_positive_min_max(dict_to_show)
    normalizer = mpl.colors.LogNorm(
        vmin=vmin,
        vmax=vmax,
        clip=False #shouldn't matter whether set to t/f
    )
    print('vmin, vmax: ', normalizer.vmin, normalizer.vmax)
    print('vmin, vmax are set? :', normalizer.scaled())


    f, ax = show_dict_with_colorbar(dict_to_show, normalizer=normalizer, cmap=cmap,
                                    **kwargs)
    if save:
        f.savefig(out_fp)
        print('saved: ', out_fp)
        
    plt.close()
    
### 
# Helpers for computing abs-diff of ffts vs. reference fft (e.g., fft of reals)
###
def compute_abs_diff(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:  #Tuple[np.ndarray, float]:
    abs_diff = np.abs(arr1-arr2)
#     l2_dist = np.sum(abs_diff ** 2)
    return abs_diff 

def compute_dict_abs_diff_vals(d: Dict[str,np.ndarray], 
                               key_of_ref_val: str) -> Dict[str, np.ndarray]:
    """computes abs-diff of each val in the dict w.r.t. dict[key_of_ref_val],
    and return the dictionary of abs-diff-vals
    """
    # first make sure d's keys are in alphabetic order
    d = dict(sorted(d.items()))

    ref_val = d[key_of_ref_val]
    return valmap(partial(compute_abs_diff, arr2=ref_val),
                          d)
    
    

# dict_abs_diff_ffts = {}
# for model_name, avg_fft in dict_avg_ffts.items():
#     dict_abs_diff_ffts[model_name] = compute_abs_diff(avg_fft, avg_fft_real)
# # store the dict as pickle:
# # joblib.dump(dict_abs_diff_ffts, "./dict-abs-diff-ffts-my-all-pass.pkl")

###
# Compute avg-magnitude-spectrum 
###
def compare_fft_options_onedir(
    img_dirpath: Path,
    out_dirpath: Path,
    n_samples: Optional[int]=None,
    kernel_size: Optional[int]=3,
    norm: Optional[str]='ortho',
):
    mkdir(out_dirpath)
    model_name = img_dirpath.name.split('-')[-1]
    n_samples = n_samples or np.inf # np.inf to use all images in the img_dirpaths
     
    f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        norm=norm
    )
    low_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: grayscale_median_filter(x, kernel_size=kernel_size),
        norm=norm
    )
    high_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: x - grayscale_median_filter(x, kernel_size=kernel_size),
        norm=norm
    )
    print('High-pass: min, max: ', high_f.min(), high_f.max())
    
    clipped_high_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: clip_floatimg(
            x - grayscale_median_filter(x, kernel_size=kernel_size)
        ),
        norm=norm
    )
    print('Clipped high-pass: min, max: ', clipped_high_f.min(), clipped_high_f.max())

    # -- Plot of all 4 types of avg-spectra in one figure
    fig, axes = show_npimgs([np.log(f), np.log(low_f), np.log(high_f), np.log(clipped_high_f)],
                titles=['f', 'low_f', 'high_f', 'clipped_high_f'],
                nrows=2)
    fig.suptitle(f'{model_name} (ks={kernel_size}, n={n_samples})')
    
    base_fn = 'fft-options'
    fig_fp = out_dirpath / f'{base_fn}_ks={kernel_size}_n={n_samples}.png'
    fig.savefig(fig_fp)
    print('Saved 4 figs: ', fig_fp )

    # -- Save only the avg-spectrum of high-pass images 
    plt.imsave( out_dirpath /f'fft-hp_ks={kernel_size}_n={n_samples}.png', 
               np.log(high_f) )
    # plt.imsave( out_dir /f'{out_fn}_avg_normed01_high_f.png', normalize_to_01(np.log(high_f) ))
    print('Done: ', model_name, end='\n')

def compute_avg_magnitude_spectrum_of_grayscale_for_each_subdirs(
    img_dir_root: Path,
    filter_type: str,
    *,
    n_samples: Optional[int]=None, 
    kernel_size: Optional[int]=3,
    norm: Optional[str]='ortho',
    save: Optional[bool]=False,
    out_dir_root: Optional[Path]=None,
    sort_dict: Optional[bool]=True,
)-> Dict[str, np.ndarray]:
    """Compute avg-fft based on images in each subdir(model_dir), and return a dictionary
    of k:v = model_name:avg_fft (np.ndarray).
    
    """
    
    img_dir_root = Path(args.img_dir_root)
    out_dir_root = Path(args.out_dir_root)
    mkdir(out_dir_root)
    
    dict_avg_ffts = {}
    for sample_dir in img_dir_root.iterdir():
        if sample_dir.is_file() or sample_dir.name.startswith("."):
            continue
        model_fullname = sample_dir.name #sample_dir.name.split('-')[-1]
        
        img_dirpath = Path(str(sample_dir.absolute()))
        out_dirpath = out_dir_root / model_fullname
        avg_fft = compute_avg_magnitude_spectrum_of_grayscale_for_onedir(
            img_dirpath=img_dirpath,
            filter_type=filter_type,
            n_samples=n_samples,
            kernel_size=kernel_size,
            norm=norm,
            save=save,
            out_dirpath=out_dirpath,
        )       
        dict_avg_ffts[model_fullname] = avg_fft
        print('Done avg-fft: ', model_fullname)
    
    # sort dict by keynames in alphabetical order
    if sort_dict:
        dict_avg_ffts = dict(sorted(dict_avg_ffts.items()))

    # save dictionary as pickle using joblib
    if save:
        # save dictionary obj (joblib)
        base_fn = f'dict-avg-ffts-{filter_type}-ks={kernel_size}-n={n_samples}-{RUN_ID}'
        out_fp = out_dir_root / f"{base_fn}.pkl"
        joblib.dump(dict_avg_ffts, out_fp)
        print('Saved dict-avg-ffts: ', out_fp)
        
        # save a plot of avg-ffts from each model( as png)
        plot_dict_spectra_logscale(dict_avg_ffts, save=True, 
                                   out_fp=out_dir_root / f"{base_fn}.png")
        
    
        
    return dict_avg_ffts
        
        
def compute_avg_magnitude_spectrum_of_grayscale_for_onedir(
    img_dirpath: Path,
    filter_type: str,
    *,
    n_samples: Optional[int]=None, 
    kernel_size: Optional[int]=3,
    norm: Optional[str]='ortho',
    save: Optional[bool]=False,
    out_dirpath: Optional[Path]=None,
) -> np.ndarray:
    """
    Args:
    ----
    img_dirpath (Path)  :path to folder containing images to compute avg. fft
    filter_type (str)   :one of allpass, highpass, lowpass to apply to each image before computing fft
    kernel_size (int)   :lowpass filter size (for media-filter) to use when filter_type is higpass or lowpass
    norm (str)          :pass to np.fft.fft2's norm argument
    n_samples (int)     :number of images to use to compute the avg. (first n_samples)
    save (bool)         :If save, save the computed np.array of averaged magnitude-spectrum of images (in grayscale)
                          in the img_dirpath, as `npy` file.
    
    Returns:
    -------
    avg_ff (np.ndarray) :computed avg-magnitude-spectrum of images in the img_dirpath (filter applied to each image)
    """
    
    model_name = img_dirpath.name.split('-')[-1]
    n_samples = n_samples or np.inf # np.inf to use all images in the img_dirpaths
    
    if filter_type == 'allpass':
        # no filtering
        f = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=n_samples,
            norm=norm
        )
    elif filter_type == 'lowpass':
        #low_f
        f = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=n_samples,
            transform=lambda x: grayscale_median_filter(x, kernel_size=kernel_size),
            norm=norm
        )
    elif filter_type == 'highpass':
        #high_f
        f = compute_avg_magnitude_spectrum_of_grayscale(
            img_dirpath,
            max_n_imgs=n_samples,
            transform=lambda x: x - grayscale_median_filter(x, kernel_size=kernel_size),
            norm=norm
        )
    else:
        raise ValueError("Unsupported filter_type: ", filter_type)
    
    print('Done computing avg mag FFT of:  ', model_name, end='\n')
    print(f'\tmin, max: ', f.min(), f.max())

    if save:
        out_dirpath = out_dirpath  or Path(f'/docker/data/GM256_avgfft_test/{model_name}') 
        mkdir(out_dirpath)
        
        filename = f'avg-fft-{filter_type}' 
        if filter_type != 'allpass':
            filename += f"_ks={kernel_size}" 
        filename += f"_n={n_samples}_{RUN_ID}" 
        # eg: avg-fft-hp_ks=3_n=1000_20230325210000.npy
        
        # np.save(out_dirpath/f"{filename}.npy", f)
        joblib.dump(f, out_dirpath/f"{filename}.pkl")
    
    return f
       
def test_compute_avg_fft_one_dir():
    pass

def test_compute_avg_fft_one_dir():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_id", required=True, help="Unique id for this run.")
    # input dir or root of input dirs
    parser.add_argument("--img_dir_root", required=False,default=None, type=str, 
                        help="Data root dir containing subdirs each of which "
                        "containing images from a GM")
    parser.add_argument("--img_dir",required=False, default=None, type=str,
                        help="Path to a folder containing images from a GM")
       

    
    # fft parameters
    parser.add_argument("--norm", required=True, type=str,
                        choices=['backward', 'ortho', 'forward'], 
                        help='Normalization mode to be used in np.fft.fft2')
    # -- to apply highpass (in pixel) before fft
    parser.add_argument("--filter_type", required=True, type=str,
                        choices=FILTER_TYPES, 
                        help="Type of filter to apply to grayscale image before FFT. "
                        "All pass applies no filtering. "
                        "For lowpass and highpass, we use median filter as low-pass "
                        "Kernel size is specified via '--kernel_size' argument.")
    parser.add_argument("-ks", "--kernel_size", default=3, type=int,
                        help='filter size for median filter for computing high-pass')
    # -- to apply pixel-value clipping (after hp) before fft
    # parser.add_argument("--plot_logscale", required=False, store-true, type=bool,
    #                     help="If set, when visualizing the avg. spectrum show it in logscale")
    # parameter for computing the average of fft's 
    parser.add_argument("-n", "--n_samples", required=False, type=int, default=None, 
                        help="Number of images to use, per imgdir, to compute"
                        " the average of spectra. If None (default), use all images in an imgdir")
    
    parser.add_argument("--save", action='store_true',  #on/off flag
                        help="If flag is specfied, save each GM's computed avg-magnitude spectrum as npy file") 
    parser.add_argument("--out_dir_root", required=False, default=None, type=str,
                        help="Root of the subdirs for fft's of images in input subdir"
                        "- Meaningful only when imgdir_root is specified.") 
    parser.add_argument("--out_dir", required=False, default=None, type=str)


# Parse cli arguments
    args = parser.parse_args()
    print('--- args ---')
    pprint(vars(args))
    exp_id = args.exp_id
    filter_type = args.filter_type.lower()
    n_samples=args.n_samples
    kernel_size = args.kernel_size
    norm = args.norm
    save = args.save
    
    if args.img_dir_root is not None:
        img_dir_root = Path(args.img_dir_root)
        
        out_dir_root = args.out_dir_root or img_dir_root.parents / 'Output-avgfft' / args.exp_id
        out_dir_root = Path(out_dir_root)
        mkdir(out_dir_root)
        
        
        dict_avg_ffts = compute_avg_magnitude_spectrum_of_grayscale_for_each_subdirs(
            img_dir_root=img_dir_root,
            filter_type=filter_type,
            n_samples=n_samples,
            kernel_size=kernel_size,
            norm=norm,
            save=save,
            out_dir_root=out_dir_root,
        )
        

        
    elif args.imgdir_root is None and args.img_dirpath is not None:
        img_dir = Path(args.img_dirpath)
        out_dir = args.out_dir or img_dir.parents / 'Output-FFT' / args.exp_id
        out_dir = Path(out_dir)
        mkdir(out_dir)
        
        avg_fft = compute_avg_magnitude_spectrum_of_grayscale_for_onedir(
            img_dirpath=img_dir,
            filter_type=filter_type,
            n_samples=n_samples,
            kernel_size=kernel_size,
            norm=norm,
            save=save,
            out_dirpath=out_dir,
        )
        
        
        
        
# run
# python compute_avg_spectrum.py -p "/data/datasets/CNN_synth_testset/stargan/1_fake/" -ks 3 -n 10
