import os
from .utils import run_bash

def convert_registration(reg_in, out_affine, reg_type=None, invert=False, overwrite=False):
    if os.path.exists(out_affine) and not overwrite:
        return out_affine
    cmd = ["mrtransform", "-linear", reg_in, out_affine]
    if invert:
        cmd += ["-inverse"]
    if reg_type:
        cmd += ["-type", reg_type]
    run_bash(cmd)
    return out_affine
