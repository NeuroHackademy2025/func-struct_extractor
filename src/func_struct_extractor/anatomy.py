import os
import nibabel as nib
from pathlib import Path
from nilearn.image import resample_to_img
from .utils import run_bash
import subprocess

def run_mrtrix_cmd(cmd, force=False):
    if force:
        cmd.append("-force")
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{e}")


def prepare_anatomy(cfg):
    sub = cfg['subject']
    fs_dir = cfg.get('fs_dir') or os.getenv('SUBJECTS_DIR')
    if not fs_dir or not os.path.isdir(os.path.join(fs_dir, sub)):
        raise FileNotFoundError(f"FreeSurfer directory for subject {sub} not found")
    fs_sub_dir = os.path.join(fs_dir, sub)

    out = {'fs_dir': fs_dir}
    fivett = cfg.get('fivett')
    fivett_out = fivett or os.path.join(cfg['output_dir'], f"{sub}_5tt.mif")
    gmwmi_out = os.path.join(cfg['output_dir'], f"{sub}_gmwmi.nii.gz")

    if not os.path.exists(fivett_out):
        run_mrtrix_cmd(['5ttgen', 'fsl', fs_sub_dir, fivett_out], force=cfg.get('overwrite', False))

    if not os.path.exists(gmwmi_out):
        run_mrtrix_cmd(['5tt2gmwmi', fivett_out, gmwmi_out], force=cfg.get('overwrite', False))

    out['5tt'] = fivett_out
    out['gmwmi'] = gmwmi_out
    return out


def prepare_rois(cfg, anat):
    rois = {}
    for roi_key in ['roi1', 'roi2']:
        if roi_key in cfg and cfg[roi_key]:
            roi_fn = cfg[roi_key]
            if not os.path.exists(roi_fn):
                raise FileNotFoundError(f"{roi_key} file not found: {roi_fn}")
            rois[roi_key] = nifti_resample_to(roi_fn, anat['gmwmi'], interp='nearest')
    return rois


def extract_bundle(cfg, anat, rois):
    tract = cfg['tract']
    roi_list = list(rois.values())
    outdir = cfg['output_dir']
    subj = cfg['subject']
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{subj}_fs_sub_bundle.tck")
    cmd = ['tckedit', tract] + roi_list + [outpath]
    run_bash(cmd)
    return outpath


def nifti_resample_to(src_img, target_img, interp='linear'):
    src = nib.load(src_img)
    tgt = nib.load(target_img)
    res = resample_to_img(src, tgt, interpolation=interp)
    fn = src_img.replace('.nii.gz', '_resampled.nii.gz')
    nib.save(res, fn)
    return fn
