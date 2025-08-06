import numpy as np
import nibabel as nib

def compute_r1(t1_path, t2_path, out_path):
    t1 = nib.load(t1_path).get_fdata().astype(np.float32)
    t2 = nib.load(t2_path).get_fdata().astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.divide(t1, t2)
        r1[~np.isfinite(r1)] = 0
    nib.save(nib.Nifti1Image(r1, affine=nib.load(t1_path).affine), out_path)
    return out_path
