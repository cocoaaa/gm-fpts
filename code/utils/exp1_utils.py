#  Define data transforms (to each feature space: RGB, Freq, SL, SSL)
import torch 
import numpy as np
from torchvision import datasets, transforms
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
    'freq_gray': xform_freq_gray,
    'freq_rgb': xform_freq_rgb,
    'freq': xform_freq,
    'rgb_freq': xform_rgb_freq,
    # 'sl': xform_sl,
    # 'ssl': xform_ssl,
}
