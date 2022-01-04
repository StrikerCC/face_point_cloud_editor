# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np

import utils
import open3d as o3
import vedo


def main():
    dicom_path = './data/722brain'
    pc_out_file_path = './model_man_face.ply'

    vol = utils.dicom_2_vol(dicom_path)
    mesh = utils.vol_2_surface(vol)
    pc = utils.mesh_2_pc(mesh)

    del vol, mesh

    pc = utils.get_rid_of_inner_points_from_pc_by_layer(pc)
    actor = o3.geometry.TriangleMesh()
    axis = actor.create_coordinate_frame(size=100)

    o3.visualization.draw_geometries([axis, pc])

    o3.io.write_point_cloud(pc_out_file_path, pc)

    # vis
    pc = vedo.load(pc_out_file_path)
    origin = vedo.Axes(xrange=(0, 100), yrange=(0, 100), zrange=(0, 100),
                       xtitle='front', ytitle='left', ztitle='head',
                       yzGrid=False, xTitleSize=0.15, yTitleSize=0.15, zTitleSize=0.15,
                       xLabelSize=0, yLabelSize=0, zLabelSize=0, tipSize=0.05,
                       axesLineWidth=2, xLineColor='dr', yLineColor='dg', zLineColor='db',
                       xTitleOffset=0.05, yTitleOffset=0.05, zTitleOffset=0.05, )

    xmin, xmax, ymin, ymax, zmin, zmax = pc.bounds()
    bound_max, bound_min = [xmax, ymax, zmax], [xmin, ymin, zmin]

    # vedo.Points.centerOfMass()
    center = pc.centerOfMass()
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]

    # bbox = vedo.Box(pos=center, length=bound_max[0] - bound_min[0], width=bound_max[1] - bound_min[1],
    #                 height=bound_max[2] - bound_min[2], alpha=0.2)
    tf = np.eye(4)

    # tf[:3, -1] = - (bound_max - bound_min) / 2
    # bbox.transform(tf)

    # vedo.show(bbox, pc)
    vedo.show(pc, origin)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
