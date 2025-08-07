import numpy as np
import nibabel as nib


def compute_approx_r1(t1_path: str, t2_path: str, nagm_mask_path: str, output_path: str) -> str:
    """
    Compute the standardized R1 (sT1w/T2w) ratio image using the formula:
        sR1 = (T1 - sT2) / (T1 + sT2), where sT2 = T2 * scale_factor
    The scale factor is the median(T1[NAGM]) / median(T2[NAGM])

    Args:
        t1_path: Path to the T1-weighted NIfTI file
        t2_path: Path to the T2-weighted NIfTI file
        nagm_mask_path: Path to the NAGM mask NIfTI file
        output_path: Path to save the output sT1w/T2w ratio NIfTI file

    Returns:
        Path to the output NIfTI file
    """
    t1_img = nib.load(t1_path)
    t2_img = nib.load(t2_path)
    mask_img = nib.load(nagm_mask_path)

    t1 = t1_img.get_fdata().astype(np.float32)
    t2 = t2_img.get_fdata().astype(np.float32)
    nagm_mask = mask_img.get_fdata().astype(bool)

    t1_nagm_median = np.median(t1[nagm_mask])
    t2_nagm_median = np.median(t2[nagm_mask])

    if t2_nagm_median == 0:
        raise ValueError("Median T2 intensity in NAGM is zero, cannot compute scaling factor.")

    scale_factor = t1_nagm_median / t2_nagm_median
    scaled_t2 = t2 * scale_factor

    numerator = t1 - scaled_t2
    denominator = t1 + scaled_t2

    with np.errstate(divide='ignore', invalid='ignore'):
        sr1 = np.true_divide(numerator, denominator)
        sr1[~np.isfinite(sr1)] = 0  

    sr1_img = nib.Nifti1Image(sr1, affine=t1_img.affine, header=t1_img.header)
    nib.save(sr1_img, output_path)

    return output_path