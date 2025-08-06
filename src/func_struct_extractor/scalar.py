import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import gaussian_weights, afq_profile
from .utils import ensure_dir

def run_scalar(cfg):
    subject = cfg['subject']
    out_base = os.path.join(cfg['output_dir'], subject)
    ensure_dir(out_base)
    tract = cfg['tract']
    scalar_cfgs = cfg['scalars']
    n_points = cfg.get('n_points', 100)

    # load streamlines
    tck = load_tractogram(tract, reference=tract, bbox_valid_check=False)
    streamlines = tck.streamlines

    weights = gaussian_weights(streamlines, n_points=n_points)

    for sc in scalar_cfgs:
        scalar_img, affine = load_nifti(sc['path'])
        # Optionally resample scalar to match tract space
        # Assuming affine shapes match for simplicity here

        profile = afq_profile(scalar_img, streamlines,
                              affine, weights, n_points=n_points)

        # Save profile and stats
        ensure_dir(out_base)
        plt.figure()
        plt.plot(profile)
        plt.ylabel(sc['name'])
        plt.xlabel("Node")
        plt.savefig(os.path.join(out_base, f"{sc['name']}_profile.png"))
        plt.close()

        stats = {
            'mean': float(np.mean(profile)),
            'std': float(np.std(profile)),
            'median': float(np.median(profile))
        }
        pd.DataFrame(stats, index=[0]).to_csv(
            os.path.join(out_base, f"{sc['name']}_stats.csv"), index=False)
