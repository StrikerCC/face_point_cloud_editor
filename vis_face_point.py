import vedo
import numpy as np
import json
from vis_reg import take_outer_surface, build_mhd, mha_to_mhd


def vis_marker_and_face(face, pts):
    """"""
    '''make marker'''
    key_pc = vedo.Points(list(pts.values()), r=15, c='r')

    '''visual'''
    vedo.show(key_pc, face)
    for key in pts.keys():
        print(key, ': ', pts[key])

        if key == 'inner_origin':
            vis_inner_frame(face)
        else:
            continue
        if len(pts[key]) < 3:
            print('ignore', key, ': ', pts[key])
            continue
        marker = vedo.Point(pos=pts[key], r=10, c='r')
        vedo.show(face, marker, title=key)

    return True


# def vis_inner_frame(face, tf=np.array([[0, -1, 0, 107.66], [0, 0, 1, 1.0], [-1, 0, 0, 236.13], [0, 0, 0, 1]])):
def vis_inner_frame(face, tf=np.array([[0, -1, 0, 107.66], [0, 0, 1, 1.0], [-1, 0, 0, 236.13], [0, 0, 0, 1]])):
    r = tf[2, 3]*1.1
    origin = vedo.Axes(xrange=(0, 100), yrange=(0, 100), zrange=(0, 100),
                       xtitle='front', ytitle='left', ztitle='head',
                       yzGrid=False, xTitleSize=0.15, yTitleSize=0.15, zTitleSize=0.15,
                       xLabelSize=0, yLabelSize=0, zLabelSize=0, tipSize=0.05,
                       axesLineWidth=2, xLineColor='dr', yLineColor='dg', zLineColor='db',
                       xTitleOffset=0.05, yTitleOffset=0.05, zTitleOffset=0.05,)
    # origin.applyTransform(np.linalg.inv(tf).tolist())
    origin.applyTransform(tf.tolist())
    # tf.squeeze()
    ball = vedo.Sphere(pos=(tf[0, -1], tf[1, -1], tf[2, -1]), r=r, alpha=0.5)

    print(tf)
    print(r)
    vedo.show(face, origin, ball)


def main():
    """"""
    ################################### ct ###################################
    print('\n################################### ct ###################################')

    '''marker data'''
    ct_face_pts_file_path = './config/facepoints_ct.json'
    f_ct = open(ct_face_pts_file_path, 'r')
    key_pts_ct = json.load(f_ct)

    '''image data'''
    # ct
    ct_file_path = 'data/model_man/brain.mhd'
    # if ct mhd not generated yet
    # dicom_path = '/home/cheng/proj/3d/TEASER-plusplus/data/human_models/head_models/model_man/722brain'
    # build_mhd(dicom_path=dicom_path, ct_file_path=ct_file_path)

    vol_ct = vedo.load(ct_file_path)
    face_ct = take_outer_surface(vol_ct, [-196.294])
    vis_marker_and_face(face_ct, key_pts_ct)

    ################################### cam ###################################
    '''marker data'''
    cam_face_pts_file_path = './config/facepoints_cam.json'
    f_cam = open(cam_face_pts_file_path)
    key_pts_cam = json.load(f_cam)

    # cam
    print('\n################################### cam ###################################')
    cam_file_path = './data/model_man/img1.0.mha'
    cam_file_path = mha_to_mhd(cam_file_path)
    vol_cam = vedo.load(cam_file_path)
    face_cam = take_outer_surface(vol_cam, [8000.0])
    vis_marker_and_face(face_cam, key_pts_cam)


if __name__ == '__main__':
    main()
