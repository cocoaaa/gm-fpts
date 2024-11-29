import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # http://matplotlib.org/faq/howto_faq.html#generate-images-without-having-a-window-appear
import matplotlib.pyplot as plt
from reprlearn.utils.image_proc import compute_avg_magnitude_spectrum_of_grayscale, clip_floatimg, grayscale_median_filter
from reprlearn.utils.image_proc import normalize_to_01
from reprlearn.visualize.utils import show_npimgs


def main(args):
    img_dirpath = Path(args.img_dirpath)
    n_samples = args.sample_size or np.inf # np.inf to use all images in the img_dirpaths
    ks = args.kernel_size
    f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
    )
    low_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: grayscale_median_filter(x, kernel_size=ks)
    )
    high_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: x - grayscale_median_filter(x, kernel_size=ks)
    )
    clipped_high_f = compute_avg_magnitude_spectrum_of_grayscale(
        img_dirpath,
        max_n_imgs=n_samples,
        transform=lambda x: clip_floatimg(
            x - grayscale_median_filter(x, kernel_size=ks)
        )
    )

    fig, axes = show_npimgs([np.log(f), np.log(low_f), np.log(high_f), np.log(clipped_high_f)],
                titles=['f', 'low_f', 'high_f', 'clipped_high_f'],
                nrows=2
            )
    fig.savefig('./out.png')

    plt.imsave('./high_f.png',np.log(high_f) )
    plt.imsave('./normed01_high_f.png',normalize_to_01(np.log(high_f) ))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--img_dirpath", required=True, type=str)
    parser.add_argument("-ks", "--kernel_size", required=True, type=int)
    parser.add_argument("-n", "--sample_size", default=None, type=int)

    args = parser.parse_args()
    main(args)

# run
# python test_compute_ffts.py -p "/data/datasets/CNN_synth_testset/stargan/1_fake/" -ks 3 -n 10

