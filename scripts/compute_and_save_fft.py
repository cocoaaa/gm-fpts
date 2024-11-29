import argparse
from typing import Callable, Optional
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib as mpl
# mpl.use('Agg') # http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
import matplotlib.pyplot as plt
from reprlearn.utils.image_proc import grayscale_median_filter, compute_magnitude_spectrum_of_grayscale_fp
from reprlearn.visualize.utils import show_npimgs
from reprlearn.utils.misc import mkdir, now2str

from IPython.core.debugger import set_trace as breakpoint

# Globals
FILTER_TYPES = ['allpass', 'highpass', 'lowpass']
FFT_NORM_TYPES = ['backward', 'ortho', 'forward']

def compute_and_save_fft(
    *,
    img_dir: Path,
    out_dir: Path,
    fft_function_on_fp: Callable,
    max_samples: Optional[int]=None, 
    out_suffix: Optional[str]='.npy', 
    is_img_fp: Optional[Callable]=None,
    **fft_kwargs,
) -> None:
    """Compute fft of each image in `img_dir` and save the fft as `npy` file
    in `out_dir`, using the original image's filename as the output filename.
    
    Args:
    ----
    img_dir            : input dir containing images to compute FFT of
    out_dir            : output dir to save each image's FFT as npy file
    fft_function_on_fp : function to apply on each img_np in img_dir to compute
        FFT
    max_samples        : number of the first N samples to compute and save FFT.
    out_format (str) : `.npy` or `.png`, but npy makes more sense.
    
    Returns:
    -------
    None
    
    """
    max_samples = max_samples or np.inf

    n_done = 0
    for img_fp in img_dir.iterdir():
        if (is_img_fp is not None) and (not is_img_fp(img_fp)):
            continue

        fp_stem = img_fp.stem #0000 [.png]
        f = fft_function_on_fp(img_fp,**fft_kwargs)
#         info(f)
        
        # scaling? -- there is no notion of "scaling" that makes sense here
        # bc what would be the max of frequency content/magnitude? 
        # so, save as npy to preserve the values (magnitude of freq. contents) as is
        out_fn = out_dir / (fp_stem + out_suffix)
        if out_suffix == ".npy":
            np.save(out_fn, f)
        elif out_suffix == ".png":
            plt.imsave(out_fn, f)
            
        n_done += 1
        
        if n_done >= max_samples:
            break
            
    print(f"Done computing fft for {n_done} imgs: {img_dir}")
    
def compute_and_save_fft_all_subdirs(
    *,
    img_dir_root: Path,
    out_dir_root: Path,
    fft_function_on_fp: Callable,
    max_samples_per_subdir: Optional[int]=None,
    is_valid_dir: Optional[Callable]=None,
):
    mkdir(out_dir_root)
    
    for img_dir in img_dir_root.iterdir():
        if (is_valid_dir is not None) and (not is_valid_dir(img_dir)): #todo
            continue
        
        dirname = img_dir.name
        out_dir = out_dir_root / dirname
        mkdir(out_dir)
        
        compute_and_save_fft(
            img_dir=img_dir,
            out_dir=out_dir,
            fft_function_on_fp=fft_function_on_fp,
            max_samples=max_samples_per_subdir
        )
        
        print('Done processing: ', img_dir)
    print('Done all img-dirs!!')


def test_compute_and_save_fft():
    exp_data_root = Path('/data/datasets/neurips23_gm256')
    img_dir = exp_data_root / 'gan-stylegan2'
    out_dir = Path('./temp')
    mkdir(out_dir)


    # run and save fft of each image in img_dir
    chosen_filter = allpass_func # choose filter
    compute_and_save_fft(
        img_dir=img_dir, 
        out_dir=out_dir,
        fft_function_on_fp=compute_magnitude_spectrum_of_grayscale_fp,
        norm='ortho', 
        transform=chosen_filter,
        max_samples=10,
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--exp_id", required=True, help="Unique id for this run."
                        " Used as outdir_root if outdir_root is None")
    # input dir or root of input dirs
    parser.add_argument("--img_dir_root", required=False, default=None, type=str, 
                        help="Data root dir containing subdirs each of which "
                        "containing images from a GM")
    parser.add_argument("--img_dir",required=False, default=None, type=str,
                        help="Path to a folder containing images from a GM")
       
    # output dir or root of output dirs
    parser.add_argument("--out_dir_root", required=False, default=None, type=str,
                        help="Root of the subdirs for fft's of images in input subdir"
                        "- Meaningful only when imgdir_root is specified.") 
    parser.add_argument("--out_dir", required=False, default=None, type=str)
    # parser.add_argument("--out_fn", required=False, default=None, type=str)
    
    # fft parameters
    parser.add_argument("--norm", required=True, type=str,
                        choices=FFT_NORM_TYPES,
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
    # parser.add_argument("--plot_logscale", required=False, store_true, type=bool,
    #                     help="If set, when visualizing the avg. spectrum show it in logscale")
    # parameter for computing the average of fft's 
    parser.add_argument("-n", "--n_samples", required=False, type=int,  default=None, 
                        help="Number of images to use, per imgdir, to compute"
                        " the average of spectra. If None (default), use all images in an imgdir")
    

    # Parse cli arguments
    args = parser.parse_args()
    exp_id = args.exp_id
    filter_type = args.filter_type.lower()
    
    # Define filter functions
    allpass_func = lambda arr: arr
    highpass_func = lambda arr: arr - grayscale_median_filter(arr, kernel_size=args.kernel_size)
    lowpass_func = lambda arr: grayscale_median_filter(arr, kernel_size=args.kernel_size)

    # choose filter to apply
    if filter_type == 'allpass':
        chosen_filter = allpass_func
    elif filter_type == 'highpass':
        chosen_filter = highpass_func
    elif filter_type == 'lowpass':
        chosen_filter = lowpass_func
        
    # Define func to apply to each img fp
    norm = args.norm.lower()
    fft_function_on_fp = partial(
        compute_magnitude_spectrum_of_grayscale_fp, 
        transform=chosen_filter,
        norm=norm,
    )

        
    if args.img_dir_root is not None:
        img_dir_root = Path(args.img_dir_root)
        
        out_dir_root = args.out_dir_root or img_dir_root.parents / 'Output-FFT' / args.exp_id
        out_dir_root = Path(out_dir_root)
        mkdir(out_dir_root)
        
        
        compute_and_save_fft_all_subdirs(
            img_dir_root=Path(args.img_dir_root),
            out_dir_root=Path(args.out_dir_root),
            fft_function_on_fp=fft_function_on_fp,
            max_samples_per_subdir=args.n_samples
            
        )
        
    elif args.imgdir_root is None and args.img_dirpath is not None:
        img_dir = Path(args.img_dirpath)
        out_dir = args.out_dir or img_dir.parents / 'Output-FFT' / args.exp_id
        out_dir = Path(out_dir)
        mkdir(out_dir)
        
        compute_and_save_fft(
            img_dir=img_dir,
            out_dir=out_dir,
            fft_function_on_fp=fft_function_on_fp,
            max_samples=args.n_samples
        )


# to run:
# conda activate test
# exp_id=20230325-174041
# img_dir_root=
# out_dir_root=
# norm=ortho
# filter_type=highpass
# n_samples=100 # don't pass it to process all images' fft's

# python compute_and_save_fft.py  --exp_id $exp_id \
#     --img_dir_root $img_dir_root --out_dir_root $out_dir_root \
#     --norm $norm  --filter_type $filter_type \
#     --n_samples $n_samples  # don't pass it to process all images' fft's
    