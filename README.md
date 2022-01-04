# Face
This is a python project build face point cloud from dicom data

## Prerequisite
python 3.8
```bash
pip install open3d==0.13.0 
pip install vedo 
pip install numpy
```

## Get started
in `main.py` change 
```python
dicom_path = './data/722brain'  # input dicom dir path 
pc_out_file_path = './model_man_face.pcd'   # output pc file path
```
to your data dir, and run `main.py`

## acknowledge
1. vedo, A python module for scientific analysis and visualization of эd objects. https://vedo.embl.es/
2. Open3D, A Modern Library for 3D Data Processing. http://www.open3d.org/
# face_point_cloud_editor
