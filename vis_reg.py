
import json
import os

import SimpleITK
import numpy as np
import open3d
import vedo
import vedo.applications
from vedo.mesh import merge
import os

def read_log(file_path):
    f = open(file_path)
    result = json.load(f)
    assert 'tsfm' in result.keys()
    return result


def build_mhd(dicom_path, ct_file_path):
    vol = vedo.load(dicom_path)
    vedo.io.write(vol, ct_file_path)
    return ct_file_path


def mha_to_mhd(file_path):
    if file_path[:-3] == 'mhd':
        return file_path
    if os.path.exists(file_path[:-3] + 'mhd'):
        return file_path[:-3] + 'mhd'
    img = SimpleITK.ReadImage(file_path)
    spacing = img.GetSpacing()
    array = SimpleITK.GetArrayFromImage(img)

    output_file = SimpleITK.GetImageFromArray(array)
    output_file.SetSpacing(spacing)
    SimpleITK.WriteImage(output_file, file_path[:-3] + 'mhd')
    return file_path[:-3] + 'mhd'


def vis_reg(reg_result, face_src, face_tgt):
    src = reg_result['src']
    tgt = reg_result['tgt']
    tracking_src = reg_result['tracking_in_src']
    tracking_tgt = reg_result['tracking_in_tgt']
    tsfm = np.asarray(reg_result['tsfm'])

    print('src', src)
    print('tgt', tgt)
    print('tracking in source', tracking_src)
    print('tracking in target', tracking_tgt)
    print('tsfm', tsfm)

    # draw points
    origin = vedo.Axes(xrange=(0, 100), yrange=(0, 100), zrange=(0, 100),
                       xtitle='front', ytitle='left', ztitle='head',
                       yzGrid=False, xTitleSize=0.15, yTitleSize=0.15, zTitleSize=0.15,
                       xLabelSize=0, yLabelSize=0, zLabelSize=0, tipSize=0.05,
                       axesLineWidth=2, xLineColor='dr', yLineColor='dg', zLineColor='db',
                       xTitleOffset=0.05, yTitleOffset=0.05, zTitleOffset=0.05,)

    pc_src = vedo.Points(src, r=18, c='g')
    pc_tgt = vedo.Points(tgt, r=18, c='r')

    arrow = vedo.Arrows(pc_src, pc_tgt, s=0.5, alpha=0.8)       # draw line between correspondence
    track_src = vedo.Arrow(tracking_src[0], tracking_src[-1], s=0.5, alpha=1.0)
    track_tgt = vedo.Arrow(tracking_tgt[0], tracking_tgt[-1], s=0.5, alpha=1.0)

    '''before registration'''
    vedo.show(pc_src, face_src, track_src, origin, title='src marker')
    vedo.show(pc_tgt, face_tgt, origin, title='tgt marker')
    vedo.show(pc_src, pc_tgt, arrow, face_tgt, origin, title='src and tgt match')

    '''after registration'''
    # apply reg result
    pc_src.applyTransform(tsfm)
    face_src = face_src.applyTransform(tsfm)
    arrow = vedo.Arrows(pc_src, pc_tgt, s=3.0, alpha=0.2)

    vedo.show(pc_src, pc_tgt, arrow, title='marker match')
    vedo.show(pc_src, pc_tgt, arrow, face_src, face_tgt, track_tgt, title='face match')
    vedo.show(face_tgt, track_tgt, title='tracking result')
    return


def take_outer_surface(file_path, threshold=[-196.294]):
    if isinstance(file_path, str):
        slice = vedo.load(file_path)
    else:
        slice = file_path
    isos = slice.isosurface(threshold=threshold)

    # splitem = isos.splitByConnectivity(maxdepth=5)[0]
    splitem = isos
    vedo.show(splitem)
    return splitem


def main():
    """"""

    ################################### cam ###################################
    '''read image'''
    cam_file_path = './data/model_man/img0.5.mhd'
    cam_file_path = mha_to_mhd(cam_file_path)
    face_src = take_outer_surface(cam_file_path, [8000.0])

    ################################### ct ###################################
    '''img file paths'''
    dicom_path = './data/model_man/722brain'
    ct_file_path = './data/model_man/brain.mhd'
    if not os.path.exists(ct_file_path):
        ct_file_path = build_mhd(dicom_path, ct_file_path)
        ct_file_path = mha_to_mhd(ct_file_path)
    face_tgt = take_outer_surface(ct_file_path, [-196.294])

    ################################### reg ###################################
    '''read reg log file'''
    log_file_path = './bin/log.json'
    reg_result = read_log(log_file_path)

    '''vis'''
    vis_reg(reg_result, face_src, face_tgt)
    return


if __name__ == '__main__':
    main()
