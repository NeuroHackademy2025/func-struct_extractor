import ants

def warp_to_mni(native_img_path, mni_template_path, output_path):
    native = ants.image_read(native_img_path)
    mni = ants.image_read(mni_template_path)
    reg = ants.registration(fixed=mni, moving=native, type_of_transform='Affine')
    warped = ants.apply_transforms(fixed=mni, moving=native, transformlist=reg['fwdtransforms'])
    ants.image_write(warped, output_path)
    return output_path
