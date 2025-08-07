import os
import subprocess
import pytest

def test_scalar_workflow(tmp_path, monkeypatch):
    config = tmp_path / "cfg.json"
    config.write_text("""
    {
      "subject": "testsub",
      "tract": "tests/data/teststreamlines.tck",
      "scalars": [
        {"path": "tests/data/test_scalar.nii.gz", "name": "TestScalar"}
      ],
      "n_points": 50,
      "output_dir": "%s/results" ,
      "overwrite": true
    }
    """ % tmp_path)
    
    cmd = f"funcstruct-extractor scalar --config {config}"
    result = subprocess.run(cmd, shell=True)
    assert result.returncode == 0
    assert (tmp_path / "results" / "testsub" / "TestScalar_profile.png").exists()

def test_extractor_workflow(tmp_path):
    config = tmp_path / "cfg.yml"
    config.write_text("""
    subject: testsub
    fs_dir: tests/data/freesurfer
    tractogram: tests/data/testbundle.trk
    roi1: tests/data/roi1.nii.gz
    roi1_name: roi1
    wmfod: tests/data/wmfod.mif
    search_dist: 2.0
    output_dir: %s/results
    overwrite: true
    """ % tmp_path)
    
    cmd = f"funcstruct-extractor extractor --config {config}"
    result = subprocess.run(cmd, shell=True)
    assert result.returncode == 0
    assert (tmp_path / "results" / "testsub_fsub_bundle.tck").exists()
