import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

def extract(subject_id, label_paths, var_paths, var_names):
    rows = []
    for label_path in label_paths:
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()
        region = Path(label_path).stem

        row = {"subject": subject_id, "region": region}
        for name, path in zip(var_names, var_paths):
            var_img = nib.load(path)
            var_data = var_img.get_fdata()
            masked = var_data[label_data > 0]
            row[name] = np.mean(masked) if masked.size else np.nan

        rows.append(row)
    return pd.DataFrame(rows)
