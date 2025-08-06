# 🧠 func_struct_extractor (trial version)

**Evolution of FSuB-extractor into a modular, flexible toolkit for neuroimaging pipelines**, integrating functional regions, diffusion tractography, ROI-based metrics, and qT₁ mapping workflows.

## 🧾 What the Package Does

- Prepares anatomical data using **FreeSurfer + MRtrix3** (5TT → GMWMI).
- Generates functional sub-bundles (**FSuBs**) via intersection of functional ROIs and tractography.
- Profiles scalar maps (e.g. FA, qT₁) along streamlines to compute tract statistics.
- Converts **volumetric labels** between native and MNI spaces.
- Extracts **ROI-based metrics** into a structured DataFrame.
- Computes **proxy R₁ maps** from T1w/T2w ratio images.
- ...

---

## 📦 Project Structure

```text
func-struct_extractor/
├── src/
│   └── func_struct_extractor/   → Core Python modules
│       ├── anatomy.py           → Anatomy prep (5tt, gmwmi, ROI resampling)
│       ├── extractor.py         → Main streamline & ROI extraction
│       ├── scalar.py            → Streamline scalar profiling
│       ├── app.py               → Click CLI entrypoint
│       ├── config.py            → Config loader/validator
│       └── ops/                 → Modular extensions
│           ├── warp_label.py              → Volumetric label → MNI warping
│           ├── extract_roi_metrics.py     → ROI scalar summary to DataFrame
│           └── compute_r1_map.py          → Compute R1 = T1w / T2w
├── tests/                       → Unit & workflow tests
│   ├── test_anatomy.py
│   └── test_workflows.py
├── README.md                    → Project documentation
└── setup.py                     → Install script (editable mode)
```

---

## 🚀 Installation

```bash
git clone git@github.com:NeuroHackademy2025/func-struct_extractor.git
cd func-struct_extractor
pip install -e src/
```

---

## 📢 CLI Commands

**Extract Functional Sub-Bundles and Scalars**

```bash
funcstruct-extractor extractor --config config_extractor.yaml
funcstruct-extractor scalar    --config config_scalar.json
```

**Volumetric Label Warp to MNI**

```bash
funcstruct-extractor warp-label \
  --native-label native_label.nii.gz \
  --t1-native native_T1.nii.gz \
  --mni-template MNI_template.nii.gz \
  --out warped_label.nii.gz
```

**Extract Metrics from ROIs**

```bash
funcstruct-extractor extract-metrics \
  --subject sub01 \
  --labels roi1.nii.gz roi2.nii.gz \
  --vars T1w.nii.gz R1_map.nii.gz \
  --var_names T1 R1 \
  --out_csv metrics_sub01.csv
```

**Compute Approximation of R₁ Map (T1w / T2w)**

```bash
funcstruct-extractor compute-r1 \
  --t1 T1w.nii.gz \
  --t2 T2w.nii.gz \
  --out R1_map.nii.gz
```

---

## 📄 Examples

- `examples/config_extractor.yaml` — for FSuB extractor
- `examples/config_scalar.json` — for scalar tractometry

---

## Summary

This toolkit now supports:

- FreeSurfer + MRtrix preprocessing
- Functional–structural bundle extraction
- Tractometry for scalar profiling
- Volumetric ROI warping + quantification
- R₁ map computation (T1w/T2w)
- ...

---