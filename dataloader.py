import numpy as np
import random
import math
import time
import struct
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import open3d as o3d
import plotly.graph_objects as go

numpoints = 20000
max_dist = 15
min_dist = 4

max_dist *= max_dist
min_dist *= min_dist

size_float = 4
size_small_int = 2

dataset_path = "dataset"

semantic_kitti_color_scheme = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    256: [255, 0, 0],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0],
}

label_remap = {
    0: 0,
    1: 0,
    10: 2,
    11: 2,
    13: 2,
    15: 2,
    16: 2,
    18: 2,
    20: 2,
    30: 2,
    31: 2,
    32: 2,
    40: 1,
    44: 1,
    48: 1,
    49: 1,
    50: 2,
    51: 2,
    52: 2,
    60: 1,
    70: 2,
    71: 2,
    72: 2,
    80: 2,
    81: 2,
    99: 2,
    252: 2,
    253: 2,
    254: 2,
    255: 2,
    256: 2,
    257: 2,
    258: 2,
    259: 2,
}

remap_color_scheme = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]

def sample(pointcloud, labels, numpoints_to_sample):
    tensor = np.concatenate((pointcloud, np.reshape(labels, (labels.shape[0], 1))), axis=1)
    tensor = np.asarray(random.choices(tensor, weights=None, cum_weights=None, k=numpoints_to_sample))
    pointcloud_ = tensor[:, 0:3]
    labels_ = tensor[:, 3]
    labels_ = np.array(labels_, dtype=np.int_)
    return pointcloud_, labels_

def readpc(pcpath, labelpath, reduced_labels=True):
    pointcloud, labels = [], []

    with open(pcpath, "rb") as pc_file, open(labelpath, "rb") as label_file:
        byte = pc_file.read(size_float * 4)
        label_byte = label_file.read(size_small_int)
        _ = label_file.read(size_small_int)

        while byte:
            x, y, z, _ = struct.unpack("ffff", byte)
            label = struct.unpack("H", label_byte)[0]

            d = x * x + y * y + z * z

            if min_dist < d < max_dist:
                pointcloud.append([x, y, z])
                if reduced_labels:
                    labels.append(label_remap[label])
                else:
                    labels.append(label)

            byte = pc_file.read(size_float * 4)
            label_byte = label_file.read(size_small_int)
            _ = label_file.read(size_small_int)

    pointcloud = np.array(pointcloud)
    labels = np.array(labels)

    return sample(pointcloud, labels, numpoints)

def remap_to_bgr(integer_labels, color_scheme):
    bgr_labels = []
    for n in integer_labels:
        bgr_labels.append(color_scheme[int(n)][::-1])
    np_bgr_labels = np.array(bgr_labels)
    return np_bgr_labels

def draw_geometries(geometries):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                                     marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=triangles[:, 0],
                                j=triangles[:, 1], k=triangles[:, 2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data'
            )
        )
    )
    fig.show()

def visualize3DPointCloud(np_pointcloud, np_labels):
    assert (len(np_pointcloud) == len(np_labels))

    pcd = o3d.geometry.PointCloud()
    v3d = o3d.utility.Vector3dVector

    pcd.points = v3d(np_pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(np_labels / 255.0)

    o3d.visualization.draw_geometries = draw_geometries

    o3d.visualization.draw_geometries([pcd])

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)

def default_transforms():
    return transforms.Compose([
        Normalize(),
        ToTensor()
    ])

class PointCloudData(Dataset):
    def __init__(self, dataset_path, transform=default_transforms(), start=0, end=1000):
        self.dataset_path = dataset_path
        self.transforms = transform

        self.pc_path = os.path.join(self.dataset_path, "sequences", "00", "velodyne")
        self.lb_path = os.path.join(self.dataset_path, "sequences", "00", "labels")

        self.pc_paths = os.listdir(self.pc_path)
        self.lb_paths = os.listdir(self.lb_path)
        assert (len(self.pc_paths) == len(self.lb_paths))

        self.start = start
        self.end = end

        self.pc_paths = self.pc_paths[start: end]
        self.lb_paths = self.lb_paths[start: end]

    def __len__(self):
        return len(self.pc_paths)

    def __getitem__(self, idx):
        item_name = str(idx + self.start).zfill(6)
        pcpath = os.path.join(self.pc_path, item_name + ".bin")
        lbpath = os.path.join(self.lb_path, item_name + ".label")

        pointcloud, labels = readpc(pcpath, lbpath)

        torch_pointcloud = torch.from_numpy(pointcloud)
        torch_labels = torch.from_numpy(labels)

        return torch_pointcloud, torch_labels

if __name__ == '__main__':
    pointcloud_index = 146
    pcpath = os.path.join(dataset_path, "sequences", "00", "velodyne", str(pointcloud_index).zfill(6) + ".bin")
    labelpath = os.path.join(dataset_path, "sequences", "00", "labels", str(pointcloud_index).zfill(6) + ".label")

    pointcloud, labels = readpc(pcpath, labelpath, False)
    labels = remap_to_bgr(labels, semantic_kitti_color_scheme)
    print("Semantic-Kitti original color scheme")
    visualize3DPointCloud(pointcloud, labels)

    pointcloud, labels = readpc(pcpath, labelpath)
    labels = remap_to_bgr(labels, remap_color_scheme)
    print("Remapped color scheme")
    visualize3DPointCloud(pointcloud, labels)
