# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 11/19/21 9:38 AM
"""
import copy

import vedo
import numpy as np
import open3d as o3


def get_rid_of_inner_all_radius(pc, center=None, vis_flag = False):
    """
    get rif of some pc inside sphere, center at input center or center of pc, radius of radius_shrink_factor * mean_norm
    of pc - max_distance_to_from_outter_layer
    :param pc:
    :type pc: open3d point cloud
    :param center:
    :type center: numpy array
    :return:
    :rtype:
    """
    radius_shrink_factor = 0.85
    max_distance_to_from_outter_layer = 1.0

    pc_return = o3.geometry.PointCloud()
    points = np.asarray(pc.points)  # (N, 3)

    points_porj_xyplane = copy.deepcopy(points)

    # mask_out = np.zeros(points_porj_xyplane.shape[0]).astype(bool)  # (N)

    if points_porj_xyplane.shape[0] < 5:
        return pc_return

    '''normalize all points: move to center, and rescale to unit length'''
    if center is None:
        center = points_porj_xyplane.mean(axis=0)
    norms = np.expand_dims(np.linalg.norm(points_porj_xyplane - center, axis=-1), axis=-1)

    '''dot product of all points, points in similar direction has dot product close to one,
    variant direction has dot product close to zero '''
    radius = norms.mean()  # (N, N)
    radius = radius_shrink_factor * radius - max_distance_to_from_outter_layer
    # print(index.shape)

    '''mark points that has similar direction'''
    mask_out = norms > radius  # (
    mask_out = mask_out.squeeze()
    '''mark points in same direction but most distant from center(having biggest norm or close to)'''

    points_out = o3.utility.Vector3dVector(points[mask_out])

    pc_return.points = points_out

    print(len(points), 'points in ')
    print(len(points_out), ' points out ')

    # debug visual
    if vis_flag:
        axis_pcd = o3.geometry.TriangleMesh()
        axis = axis_pcd.create_coordinate_frame(size=100, origin=center)
        sphere_filter = axis_pcd.create_sphere(radius=radius, resolution=1000)
        sphere_filter.translate(center)
        sphere_filter = sphere_filter.sample_points_uniformly(number_of_points=1000)

        o3.visualization.draw_geometries([pc, sphere_filter, axis])
        o3.visualization.draw_geometries([pc_return])

    return pc_return


def get_rid_of_inner_points_from_pc_by_layer(pc):
    """
    get rid of points inside head point cloud
    :param pc: head pc
    :type pc: open3d point cloud
    :return: face pc
    :rtype: open3d point cloud
    """
    layer_axis = 2
    layer_step = 35

    max_bound_org, min_bound_org = o3.geometry.PointCloud.get_max_bound(pc), o3.geometry.PointCloud.get_min_bound(pc)
    center_org = pc.get_center()
    pc_deform = copy.deepcopy(pc)

    '''cut off part below neck'''
    # bounding_box = o3.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    # pc_deform = o3.geometry.PointCloud.crop(pc_deform, bounding_box)

    '''normalized and scale head toward sphere shape'''
    # sphere_radius = 100.0
    points = np.array(pc_deform.points)
    points -= center_org

    ratio = np.array([100.0, 100.0, 200.0]) / (max_bound_org - center_org)
    scale_matrix = np.eye(3)
    scale_matrix[0, 0] = ratio[0]
    scale_matrix[1, 1] = ratio[1]
    scale_matrix[2, 2] = ratio[2]
    points = np.matmul(points, scale_matrix)

    # points += center
    pc_deform.points = o3.utility.Vector3dVector(points)

    # new range
    max_bound_deformed, min_bound_deformed = o3.geometry.PointCloud.get_max_bound(pc_deform), o3.geometry.PointCloud.get_min_bound(pc_deform)

    # vis
    o3.visualization.draw_geometries([pc_deform])

    pc_return = o3.geometry.PointCloud()

    '''filter points inside layer-wise'''
    for slice_start_in_layer_axis in np.arange(min_bound_deformed[layer_axis], max_bound_deformed[layer_axis], layer_step):
        '''crop layer point cloud'''
        max_bound_layer, min_bound_layer = copy.deepcopy(max_bound_deformed), copy.deepcopy(min_bound_deformed)
        max_bound_layer[layer_axis] = slice_start_in_layer_axis + layer_step
        min_bound_layer[layer_axis] = slice_start_in_layer_axis
        bounding_box = o3.geometry.AxisAlignedBoundingBox(min_bound_layer, max_bound_layer)
        layer = o3.geometry.PointCloud.crop(pc_deform, bounding_box)

        # o3.visualization.draw_geometries([layer])

        '''layer points '''
        pc_return += get_rid_of_inner_all_radius(layer, center=(0, 0, 0))

    print('total ', len(np.array(pc_deform.points)), ' in input')
    print('total ', len(np.array(pc_return.points)), ' in output')

    '''scale back to original'''
    points_undeformed = np.matmul(np.asarray(pc_return.points), np.linalg.inv(scale_matrix))
    points_undeformed += center_org
    pc_return.points = o3.utility.Vector3dVector(points_undeformed)
    return pc_return


def mesh_2_pc(mesh):
    points = mesh.points()
    pc = o3.geometry.PointCloud()
    pc.points = o3.utility.Vector3dVector(points)
    return pc


def vol_2_surface(vol, intensity=[-196.294]):
    """
    volume to connected surface with same intensity
    :param vol: volume data
    :type vol: vedo.Volume
    :param intensity: list of intensity
    :type intensity: list
    :return: iso-surface
    :rtype: vedo.mesh
    """
    isosurfaces = []
    '''crop neck'''
    vol.crop(left=0.0, right=0.0, back=0.0, front=0.0,
             bottom=0.0, top=0.25)
    vedo.show(vol)

    spacing = vol.spacing()
    print('dicom Spacing: ', spacing)
    # vedo.show(vol, axes=1)

    '''build surface layer by layer'''
    slice_step = 0.1
    for frac_slice_bottom in np.arange(0.0, 1.0, slice_step):
        frac_slice_top = 1.0 - frac_slice_bottom - slice_step
        print('frac start at ', frac_slice_bottom)
        # slice = vol.copy()
        slice = vol.clone()
        slice.crop(left=0.0, right=0.0, back=0.0, front=0.0,
                   bottom=frac_slice_bottom, top=frac_slice_top)
        threshold = intensity
        isos = slice.isosurface(threshold=threshold)
        splitems = isos.splitByConnectivity(maxdepth=5)

        '''take the most outer surface'''
        radius_biggest, splitem_farest = 0, splitems[0]
        for splitem in splitems:
            points = splitem.points()
            radius = np.linalg.norm(points - points.mean(axis=0), axis=-1).mean(axis=0)
            if radius_biggest > radius:
                radius_biggest = radius
                splitem_farest = splitem
        isosurfaces.append(splitem_farest)

        '''vis current slice'''
        # vedo.show(splitem_farest)

    face = vedo.merge(isosurfaces)

    vedo.show(face)

    return face


def face_from_seg_label(file_path='./data/', label=2):
    pass


def dicom_2_vol(file_path):
    """

    :param file_path: dicom dir path
    :type file_path: str
    :return: volume data
    :rtype: vedo.Volume
    """
    vol = vedo.load(file_path)
    vedo.show(vol)
    return vol


def bbox(max_bound, min_bound):
    x_max, y_max, z_max = max_bound
    x_min, y_min, z_min = min_bound
    p111 = [x_max, y_max, z_max]
    p011 = [x_min, y_max, z_max]
    p101 = [x_max, y_min, z_max]
    p110 = [x_max, y_max, z_min]

    p100 = [x_max, y_min, z_min]
    p010 = [x_min, y_max, z_min]
    p001 = [x_min, y_min, z_max]
    p000 = [x_min, y_min, z_min]

    points = [p111]
    return
