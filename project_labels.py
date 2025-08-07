import numpy as np
import neuropythy as ny
import ants
import nibabel as nib
import mne

def surf_label_2_vol(sub_id, label, fs_dir, out_dir=None, hemi='both'):
    '''
    Convert surface labels to volumetric NIfTI images for specified subjects and ROIs.

    Parameters:
    sub_id (str): A list of subject IDs or a single subject ID.
    labels (str): A list of labels or a single label.
    fs_dir (str): Directory containing FreeSurfer subjects.
    out_dir (str or None): Directory to save the output NIfTI volumes. If None, volumes are not saved. Default is None.
    hemi (str): Specify 'lh', 'rh', or 'both' to determine which hemispheres to process. Default is 'both'.

    Returns:
    np.ndarray: An array of shape (number of subjects, number of ROIs) containing the generated volumes.
    '''
  
    # Make sure that the ny module is ready to be used
    sub = ny.freesurfer_subject(f'{fs_dir}{sub_id}/')

    # Create a template volume to fill with labels
    template = sub.images['ribbon']
    template = ny.image_clear(template)
    
    # Initialize masks for both hemispheres
    label_lh_mask = np.zeros(sub.lh.vertex_count)
    label_rh_mask = np.zeros(sub.rh.vertex_count)

    # Process left hemisphere if needed
    if hemi in ['lh', 'both']:
        label_lh = sub.load(f'label/lh.{label}.label')
        label_lh_mask[label_lh[0]] = True

    # Process right hemisphere if needed
    if hemi in ['rh', 'both']:
        label_rh = sub.load(f'label/rh.{label}.label')
        label_rh_mask[label_rh[0]] = True

    # Convert masks to NIfTI object
    vol = sub.cortex_to_image((label_lh_mask, label_rh_mask), im=template)

    # Save volume if out_dir is provided
    if out_dir is not None:
        vol.to_filename(f'{out_dir}{sub_id}_{label}_vol.nii.gz')
    
    return vol


def vol_label_2_surf(sub_id, label, fs_dir, out_dir=None, hemi='both'):
    '''
    Convert surface labels to volumetric NIfTI images for specified subjects and ROIs.

    Parameters:
    sub_id (str): A single subject ID.
    labels (nii object): A nifti object for a single label.
    fs_dir (str): Directory containing FreeSurfer subjects.
    out_dir (str or None): Directory to save the output NIfTI volumes. If None, volumes are not saved. Default is None.
    hemi (str): Specify 'lh', 'rh', or 'both' to determine which hemispheres to process. Default is 'both'.

    Returns:
    np.ndarray: An array of shape (number of subjects, number of ROIs) containing the generated volumes.
    '''
  
    # Make sure that the ny module is ready to be used
    sub = ny.freesurfer_subject(f'{fs_dir}{sub_id}/')

    # Convert masks to NIfTI object
    lh_label, rh_label = sub.image_to_cortex(label)

    lh_label_indices = np.where(lh_label > 0)[0]
    if len(lh_label_indices) > 0:
        left_label = mne.Label(lh_label_indices, hemi='lh')
        if out_dir is not None:
            left_label.save(f'{out_dir}lh_{sub_id}_{label}')    
    else:
        return('no vertices in left hemisphere')

    rh_label_indices = np.where(rh_label > 0)[0]
    if len(lh_label_indices) > 0:
        right_label = mne.Label(rh_label_indices, hemi='lh')
        if out_dir is not None:
                    right_label.save(f'{out_dir}rh_{sub_id}_{label}')    else:
        return('no vertices in right hemisphere')
            
    return (left_label, right_label)

    
def fsnative_label_2_fsaverage(fsnative_path, fsaverage_path, fsnative_label,
                                hemisphere, output_filename=None):
    '''
    Interpolate a surface label from a native FreeSurfer subject to the fsaverage surface.

    This function loads a native surface label, interpolates it onto the fsaverage surface,
    and optionally saves the resulting label as a .label file.

    Parameters:
    fsnative_path (str): The file path to the native surface mesh (e.g., `.pial`).
    fsaverage_path (str): The file path to the fsaverage surface mesh (e.g., `.pial`).
    fsnative_label (array): The array contains the vertices of the native surface label.
    hemisphere (str): The hemisphere to process ('lh' for left hemisphere or 'rh' for right hemisphere).
    output_filename (str or None): The filename to save the label. 
                                    If provided, the label will be saved; if None, it will not be saved.

    Returns:
    mne.Label: The interpolated label for the specified hemisphere.
    '''
    
    # Load FreeSurfer subjects
    fsn_sub = ny.freesurfer_subject(fsnative_path)
    fsa_sub = ny.freesurfer_subject(fsaverage_path)

    # Interpolate the label to the fsaverage surface
    if hemisphere == 'lh':
        # Create a boolean array for the label with the same vertex count as the subject's surface
        fsn_label = np.zeros(fsn_sub.lh.vertex_count, dtype=bool)
    
        # Set the specified indices to True
        fsn_label[fsnative_label] = True
        
        fsa_label = fsn_sub.lh.interpolate(fsa_sub.lh, fsn_label)
    elif hemisphere == 'rh':
        # Create a boolean array for the label with the same vertex count as the subject's surface
        fsn_label = np.zeros(fsn_sub.rh.vertex_count, dtype=bool)
    
        # Set the specified indices to True
        fsn_label[fsnative_label] = True
        
        fsa_label = fsn_sub.rh.interpolate(fsa_sub.rh, fsn_label)
    else:
        raise ValueError("Hemisphere needs to be specified as 'lh' or 'rh'.")

    # Find the indices of the interpolated labels that are True
    fsa_label_indices = np.where(fsa_label == True)[0]

    # Create an MNE Label object using the interpolated indices
    label = mne.Label(fsa_label_indices, hemi=hemisphere)

    # Save the label if an output filename is provided
    if output_filename is not None:
        label.save(output_filename)

    return label

def fsaverage_label_2_fsnative(fsaverage_path, fsnative_path, fsaverage_label,
                                hemisphere, output_filename=None):
    '''
    Interpolate a surface label from a native FreeSurfer subject to the fsaverage surface.

    This function loads a native surface label, interpolates it onto the fsaverage surface,
    and optionally saves the resulting label as a .label file.

    Parameters:
    fsaverage_path (str): The file path to the fsaverage surface mesh (e.g., `.pial`).
    fsnative_path (str): The file path to the native surface mesh (e.g., `.pial`).
    fsaverage_label (array): The array contains the vertices of the average surface label.
    hemisphere (str): The hemisphere to process ('lh' for left hemisphere or 'rh' for right hemisphere).
    output_filename (str or None): The filename to save the label. 
                                    If provided, the label will be saved; if None, it will not be saved.

    Returns:
    mne.Label: The interpolated label for the specified hemisphere.
    '''
    
    # Load FreeSurfer subjects
    fsa_sub = ny.freesurfer_subject(fsaverage_path)
    fsn_sub = ny.freesurfer_subject(fsnative_path)

    # Interpolate the label to the fsaverage surface
    if hemisphere == 'lh':
        # Create a boolean array for the label with the same vertex count as the subject's surface
        fsa_label = np.zeros(fsa_sub.lh.vertex_count, dtype=bool)
    
        # Set the specified indices to True
        fsa_label[fsaverage_label] = True
        
        fsn_label = fsa_sub.lh.interpolate(fsn_sub.lh, fsa_label)
    elif hemisphere == 'rh':
        # Create a boolean array for the label with the same vertex count as the subject's surface
        fsa_label = np.zeros(fsa_sub.rh.vertex_count, dtype=bool)
    
        # Set the specified indices to True
        fsa_label[fsaverage_label] = True
        
        fsn_label = fsa_sub.rh.interpolate(fsn_sub.rh, fsa_label)
    else:
        raise ValueError("Hemisphere needs to be specified as 'lh' or 'rh'.")

    # Find the indices of the interpolated labels that are True
    fsn_label_indices = np.where(fsn_label == True)[0]

    # Create an MNE Label object using the interpolated indices
    label = mne.Label(fsn_label_indices, hemi=hemisphere)

    # Save the label if an output filename is provided
    if output_filename is not None:
        label.save(output_filename)

    return label


def MNI_label_2_native(fs_dir:str, sub_id: str, mni_fname: str):

    '''
    Convert MNI labels to native volumetric space.

    Parameters:
    fs_dir (str): The path to the FS directory. 
    sub (str): The subject ID as string. 
    mni_fname (str): The path to the mni space image. 
    
    Returns:
    mni_SyN_trans: the volumetric rois in native space.
    
    '''

    print(f'Converting MNI ROI labels to Native space for {sub_id}')
    
    # Retrieve the T1 image from subject freesurfer directory
    t1_fname = nsd_base_path / 'nsddata' / 'ppdata' / sub_id  /'anat' / 'T1_1pt0_masked.nii.gz'
    t1_img = ants.image_read(t1_fname.fspath)

    t1_brain_mask = ants.get_mask(image = t1_img, 
                                  low_thresh = 500, high_thresh = 2000, 
                                  cleanup = 2)

    mni_template = ants.image_read(ants.get_data('mni'))

    # Get the MNI image from ANTS
    if mni_fname: 
        print('Using provided MNI ROI Mask Definition')
        mni_img = ants.image_read(mni_fname)       

    print('Calculating Affine Transformation')
        
    # Calculate the Affine Transformation
    affine_reg = ants.registration(fixed = t1_brain_mask, 
                                   moving = mni_template,
                                   # mask = t1_brain_mask,
                                   moving_mask = mni_img,  
                                   type_of_transform = 'Affine')

    # Apply Affine Transformation 
    mni_affine_trans = ants.apply_transforms(fixed = t1_brain_mask, 
                                             moving = mni_img,
                                             transformlist=affine_reg['fwdtransforms'])
    print('Calculating SyN Transformation')
    # Calculate SyN Transformation
    SyN_reg = ants.registration(fixed = t1_brain_mask, 
                                # mask = t1_brain_mask,
                                moving = mni_affine_trans,
                                type_of_transform = 'SyN')

    # Apply SyN Transformation
    mni_SyN_trans = ants.apply_transforms(fixed = t1_brain_mask, 
                                          moving = mni_affine_trans, 
                                          transformlist = SyN_reg['fwdtransforms'])

    mni_SyN_trans.image_write(filename = f'/home/jovyan/projects/func-struct_extractor/data/derivatives/{sub_id}_MNI_to_Native.nii.gz')

    print('Plotting Coregistration')
    # Plot the coregistration and save it in a derivative data directory
    ants.plot(image = t1_img, 
              overlay = mni_SyN_trans, 
              overlay_alpha = 0.5, 
              filename=f'/home/jovyan/projects/func-struct_extractor/data/derivatives/coregistration/{sub_id}_registration_plot.png', 
              title = f'{sub_id} MNI to Native Coregistration')
    
    print(f'{sub_id} Finished.\n')
    
    # Return the native volumetric space
    return mni_SyN_trans