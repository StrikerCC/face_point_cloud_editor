# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/3/21 4:54 PM
"""
import sys

import numpy as np
import open3d as o3
import json

sys.path.append('/home/cheng/proj/3d/TEASER-plusplus/')
from registration import ransac_icp_helper

holes_cam = {
    'left_front': [],
    'left_mid': [],
    'right_front': [[-13.2, -109.31, 579.89],
                    [-12.51, -102.52, 589.5],
                    [-20.01, -95.74, 585.87],
                    [-20.42, -102.92, 577.05]],
    'right_mid': [[33.86, -134.55, 599.22],
                  [26.07, -126.76, 609.86],
                  [20.26, -126.86, 601.24]]
}


def main():
    """"""
    pc_inferred_cam_file_path = '../3D_model_infered_camera.pcd'
    pc_ct_file_path = '../model_man_face.ply'
    '''compute tf from inferred cam to '''
    face_inferred_cam = o3.io.read_point_cloud(pc_inferred_cam_file_path)
    face_ct = o3.io.read_point_cloud(pc_ct_file_path)

    result_global, result_local, time_global, time_local = ransac_icp_helper(pc_src=face_inferred_cam, pc_tgt=face_ct,
                                                                             voxel_size_global=5,
                                                                             voxel_sizes_local=1)
    tf_cam_2_ct = result_local.transformation
    # print(tf_global)
    print(tf_cam_2_ct)

    '''compute center of hole'''
    holes_np_cam = {}
    holes_np_ct = {}
    for hole_name in holes_cam.keys():
        if len(holes_cam[hole_name]) > 0:
            holes_np_cam[hole_name] = np.mean(holes_cam[hole_name], axis=0)
            holes_np_cam[hole_name] = np.concatenate([holes_np_cam[hole_name], np.ones(1)])
    print(holes_np_cam)

    '''apply tf to points in inferred cam frame'''
    for hole_name in holes_np_cam.keys():
        holes_np_ct[hole_name] = np.dot(tf_cam_2_ct, holes_np_cam[hole_name])
        holes_np_ct[hole_name] = holes_np_ct.get(hole_name)[0:3].tolist()
    print(holes_np_ct)

    '''vis debug'''
    mesh_ = o3.geometry.TriangleMesh()
    right_mid = mesh_.create_coordinate_frame(size=100.0, origin=holes_np_ct['right_mid'])
    # right_mid = mesh_.create_cone(radius=5.0, height=5.0)
    # right_mid.apply
    o3.visualization.draw_geometries([face_ct, right_mid])
    '''save result'''
    f_cam = open('./holes_infredcam.json', 'w')

    # json.dump(holes_cam, f_cam)


if __name__ == '__main__':
    main()
