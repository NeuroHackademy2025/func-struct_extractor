# 🧠 func_struct_extractor 

A set of Python scripts for working with **surface** and **volumetric** neuroimaging ROIs, built on FreeSurfer and NIfTI formats.
These scripts implement core transformations between spaces and formats to support flexible ROI usage in structural and functional MRI analysis.


## 🧾 Features

**ROI Projection**
- Surface → Volume
- Volume → Surface
- Template ↔ Native space

**Label Conversions**
- NIfTI binary masks ↔ FreeSurfer label files
- FSaverage ↔ FSnative label interpolation (neuropythy)

**Spatial Registration**
- MNI ↔ Native space via affine and SyN (non-linear) registration
- Handles both volumetric and surface-based ROIs

