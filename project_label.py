import neuropythy as ny
import numpy as np
from nilearn import surface
import mne
import nibabel as nib
import project_label_utilities as pl


def project_label(sub_id, label, fs_dir, out_dir, 
                   from_space='surface', to_space='volumetric', 
                   fsnative_path=None, fsaverage_path=None, 
                   t1_fname=None, roi_fname=None, calc_brain_mask=False, save_coreg=False):
    """
    Project labels from one space to another (surface to volumetric or vice versa).

    Parameters:
    -----------
    sub_id (str): The subject ID.
    label (str or numpy.ndarray): The label(s) to project.
    fs_dir (str): Directory containing FreeSurfer subjects.
    out_dir (str): Directory to save output labels.
    from_space (str): The space of the input label ('surface', 'volumetric', 'fsnative', 'fsaverage', 'MNI', 'native').
    to_space (str): The space to which to convert ('surface', 'volumetric', 'fsnative', 'fsaverage', 'MNI', 'native').
    fsnative_path (str): Path to the native FreeSurfer subject or mesh.
    fsaverage_path (str): Path to the fsaverage surface.
    t1_fname (str): Path to the T1 file for MNI to Native conversion.
    roi_fname (str): Path to the ROI file for label conversion.
    calc_brain_mask (bool): Whether to calculate a brain mask.
    save_coreg (bool): Whether to save the coregistration to file.

    Returns:
    --------
    dict or mne.Label or ants.ANTsImage: The projected label(s) in the new space.
    """
    
    if from_space == 'surface' and to_space == 'volumetric':
        # Convert surface label to volumetric
        vol_output = pl.surf_label_2_vol(sub_id, label, fs_dir, roi_fname, out_dir=out_dir)
        return vol_output
    
    elif from_space == 'volumetric' and to_space == 'surface':
        print(out_dir)
        # Convert volumetric label to surface
        vol_label = nib.load(label)  # Assuming `label` is a filename for a volumetric NIfTI image.
        surf_output = pl.vol_label_2_surf(sub_id, vol_label, fs_dir, roi_fname, out_dir)
        return surf_output  # Now returns a dictionary with 'lh_label' or 'rh_label'

    elif from_space == 'fsnative' and to_space == 'fsaverage':
        if fsnative_path is None or fsaverage_path is None:
            raise ValueError("Paths for fsnative and fsaverage must be provided.")
        
        native_label_indices = label  # Assuming `label` is an array of vertex indices.
        surf_output = pl.fsnative_label_2_fsaverage(fsnative_path, fsaverage_path, native_label_indices, 'lh', out_dir)
        return surf_output
    
    elif from_space == 'fsaverage' and to_space == 'fsnative':
        if fsnative_path is None or fsaverage_path is None:
            raise ValueError("Paths for fsnative and fsaverage must be provided.")
        
        average_label_indices = label  # Assuming `label` is an array of vertex indices.
        native_output = pl.fsaverage_label_2_fsnative(fsaverage_path, fsnative_path, average_label_indices, 'lh', out_dir)
        return native_output

    elif from_space == 'MNI' and to_space == 'native':
        if t1_fname is None or roi_fname is None:
            raise ValueError("Paths for T1 file and ROI file must be provided.")
        
        native_output = pl.MNI_label_2_native(t1_fname, sub_id, calc_brain_mask, roi_fname, save_coreg, out_dir)
        return native_output
    
    elif from_space == 'native' and to_space == 'MNI':
        if t1_fname is None or roi_fname is None:
            raise ValueError("Paths for T1 file and ROI file must be provided.")
            
        mni_output = pl.native_label_2_mni(t1_fname, sub_id, calc_brain_mask, roi_fname, save_coreg, out_fname)
        return mni_output
    
    else:
        raise ValueError("Unsupported space conversion requested.")