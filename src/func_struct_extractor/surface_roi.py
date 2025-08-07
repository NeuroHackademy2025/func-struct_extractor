import os
import nibabel as nib
from nilearn.surface import load_surf_data
from .utils import nibabel_label_to_volume

def surface_to_volume(hemi_label, fs_dir, subject, target_img, out_vol):
    label = load_surf_data(hemi_label)
    vol = nibabel_label_to_volume(label, subject, fs_dir)
    nib.save(vol, out_vol)
    return out_vol
