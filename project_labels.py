import numpy as np
import neuropythy as ny

def surf_label_2_vol(subs, ROIs, fs_dir, out_dir=None, hemispheres='both'):
    '''
    Convert surface labels to volumetric NIfTI images for specified subjects and ROIs.

    Parameters:
    subs (list or str): A list of subject IDs or a single subject ID.
    ROIs (list or str): A list of ROIs or a single ROI.
    fs_dir (str): Directory containing FreeSurfer subjects.
    out_dir (str or None): Directory to save the output NIfTI volumes. If None, volumes are not saved. Default is None.
    hemispheres (str): Specify 'lh', 'rh', or 'both' to determine which hemispheres to process. Default is 'both'.

    Returns:
    np.ndarray: An array of shape (number of subjects, number of ROIs) containing the generated volumes.
    '''
    
    if isinstance(subs, str):
        subs = [subs]
    
    if isinstance(ROIs, str):
        ROIs = [ROIs]
    
    vol_array = np.empty((len(subs), len(ROIs)), dtype=object)
    
    for i, sub_id in enumerate(subs):
        # Make sure that the ny module is ready to be used
        sub = ny.freesurfer_subject(f'{fs_dir}{sub_id}/')

        for j, roi in enumerate(ROIs):
            # Create a template volume to fill with labels
            template = sub.images['ribbon']
            template = ny.image_clear(template)
            
            # Initialize masks for both hemispheres
            label_lh_mask = np.zeros(sub.lh.vertex_count)
            label_rh_mask = np.zeros(sub.rh.vertex_count)

            # Process left hemisphere if needed
            if hemispheres in ['lh', 'both']:
                label_lh = sub.load(f'label/lh.{roi}.label')
                label_lh_mask[label_lh[0]] = True

            # Process right hemisphere if needed
            if hemispheres in ['rh', 'both']:
                label_rh = sub.load(f'label/rh.{roi}.label')
                label_rh_mask[label_rh[0]] = True

            # Convert masks to NIfTI objects (the other hemisphere will just be zeros)
            vol = sub.cortex_to_image((label_lh_mask, label_rh_mask), im=template)

            # Save volume if out_dir is provided
            if out_dir is not None:
                vol.to_filename(f'{out_dir}{sub_id}_{roi}_vol.nii.gz')

            # Store the volume for this subject and ROI
            vol_array[i, j] = vol
            
    return vol_array