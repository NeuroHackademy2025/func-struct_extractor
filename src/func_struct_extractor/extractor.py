import os
from .anatomy import prepare_anatomy, prepare_rois, extract_bundle
from .utils import ensure_dir

def run_extractor(cfg):
    subject = cfg['subject']
    out_base = cfg['output_dir']
    ensure_dir(out_base)

    anat = prepare_anatomy(cfg)
    rois = prepare_rois(cfg, anat)
    tract = extract_bundle(cfg, anat, rois)

    print(f"[✅] Func-struct completed. Tract output: {tract}")
