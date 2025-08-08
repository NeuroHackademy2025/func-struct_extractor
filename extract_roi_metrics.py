import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple

def load_nifti(path: Union[str, Path]) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img

def check_affines(ref_img: nib.Nifti1Image, other_imgs: List[nib.Nifti1Image]):
    ref_affine = ref_img.affine
    for img in other_imgs:
        if not np.allclose(ref_affine, img.affine):
            raise ValueError("Affine mismatch: all images must be in the same space.")

def extract_metrics_from_roi(
    label_img_path: Union[str, Path],
    metric_img_paths: List[Union[str, Path]],
    metric_names: List[str],
    subject_id: str
) -> pd.DataFrame:
    """
    Extract mean scalar metrics for each labeled region from volumetric ROIs.

    Parameters:
        label_img_path: NIfTI image with integer labels per voxel (atlas or ROI)
        metric_img_paths: List of paths to scalar images (e.g., T1, qT1, FA)
        metric_names: Names of the scalar variables to assign to columns
        subject_id: Subject identifier

    Returns:
        pd.DataFrame with columns: ['subject', 'region'] + metric_names
    """
    label_data, label_img = load_nifti(label_img_path)
    metric_data_list = [load_nifti(p)[0] for p in metric_img_paths]
    metric_imgs = [nib.load(str(p)) for p in metric_img_paths]
    
    # Ensure all images are in the same space
    check_affines(label_img, metric_imgs)

    df_rows = []
    region_ids = np.unique(label_data)
    region_ids = region_ids[region_ids != 0]  # Exclude background (0)

    for region_id in region_ids:
        mask = label_data == region_id
        region_label = f"region_{int(region_id)}"

        metrics = []
        for metric_data in metric_data_list:
            masked_values = metric_data[mask]
            mean_val = np.nanmean(masked_values)
            metrics.append(mean_val)

        row = [subject_id, region_label] + metrics
        df_rows.append(row)

    columns = ['subject', 'region'] + metric_names
    return pd.DataFrame(df_rows, columns=columns)

