import pytest
import tempfile
import shutil
import nibabel as nib
import numpy as np
from pathlib import Path
from func_struct_extractor import anatomy

@pytest.fixture
def dummy_cfg():
    tmp_dir = tempfile.mkdtemp()
    fs_dir = Path(tmp_dir) / "freesurfer"
    subj_dir = fs_dir / "sub-test"
    subj_dir.mkdir(parents=True)

    return {
        'subject': 'sub-test',
        'fs_dir': str(fs_dir),
        'output_dir': tmp_dir
    }

@pytest.fixture
def dummy_nii():
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    data = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, tmp.name)
    return tmp.name

def test_nifti_resample_to(dummy_nii):
    # Resample dummy file to itself, should not raise
    out = anatomy.nifti_resample_to(dummy_nii, dummy_nii)
    assert Path(out).exists()
    img = nib.load(out)
    assert img.shape == (10, 10, 10)

def test_prepare_rois_resampling(dummy_cfg, dummy_nii):
    dummy_cfg['roi1'] = dummy_nii
    dummy_cfg['roi2'] = dummy_nii
    anat = {'gmwmi': dummy_nii}
    rois = anatomy.prepare_rois(dummy_cfg, anat)
    assert 'roi1' in rois and Path(rois['roi1']).exists()
    assert 'roi2' in rois and Path(rois['roi2']).exists()
