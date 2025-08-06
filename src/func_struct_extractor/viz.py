import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.plotting import plot_anat, plot_roi
from dipy.viz import window, actor

def plot_slice_overlay(anat_img, scalar_img, stat='median', output_png=None):
    stat_img = nib.load(scalar_img)
    fig = plot_anat(stat_img, display_mode='ortho', annotate=False)
    if output_png:
        fig.savefig(output_png)
    return fig

def viz_tract_with_roi(tract_file, roi_file, background_img=None, fname=None):
    sl = window.Scene()
    tract = actor.line(load_tracks(tract_file), color=(1,0,0))
    sl.add(tract)
    if roi_file:
        roi = actor.contour_from_roi(roi_file, color=(0,1,0))
        sl.add(roi)
    if background_img:
        bg = actor.slicer(nib.load(background_img).get_fdata())
        sl.add(bg)
    window.record(sl, out_path=fname, size=(800,800))
