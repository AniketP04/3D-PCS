import os 
import math 
import random 
import numpy as np 

import torch 
from torchvision import transforms


def load_data(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_vertices, n_faces, _ = map(int, file.readline().strip().split())
    vertices = [list(map(float, file.readline().strip().split())) for _ in range(n_vertices)]
    faces = [list(map(int, file.readline().strip().split()[1:])) for _ in range(n_faces)]
    return vertices, faces 


class PointSampler(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        s, t = sorted([random.random(), random.random()])
        f = lambda x: s * pt1[x] + (t-s) * pt2[x] + (1-t) * pt3[x]
        return [f(0), f(1), f(2)]

    def __call__(self, mesh):
        vertices, faces = mesh
        vertices = np.array(vertices)
        areas = [self.triangle_area(vertices[faces[i][0]], vertices[faces[i][1]], vertices[faces[i][2]])
                  for i in range(len(faces))]
        face_samples = random.choices(faces, weights = areas, k = self.output_size)
        point_samples = np.array([self.sample_point(vertices[face_samples[i][0]], vertices[face_samples[i][1]], vertices[face_samples[i][2]])
                          for i in range(len(face_samples))])
        return point_samples


class Normalize(object):
    def __call__(self, pointcloud):
        norm_pointcloud = (pointcloud - np.mean(pointcloud, axis=0))
        norm_pointcloud /= max(np.linalg.norm(pointcloud, axis=1))
        return norm_pointcloud


def default_transforms():
    return transforms.Compose([
        PointSampler(1024),
        Normalize()
    ])


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms(), data_augmentation=True):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {f: i for i, f in enumerate(folders)}
        self.transform = transform 
        self.data_augmentation = data_augmentation 
        self.files = []
        for category in self.classes.keys(): 
            new_dir = root_dir/category/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)
        
    def __getitem__(self, idx):
        path = self.files[idx]['path']
        category = self.files[idx]['category']
        with open(path, 'r') as f: 
            vertices, faces = load_data(f)
            pointcloud = self.transform((vertices, faces))
        
        if self.data_augmentation: 
            theta = random.random() * 2 * math.pi
            rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0], 
                                   [math.sin(theta),  math.cos(theta), 0], 
                                   [0,                0,               1]])
            pointcloud = np.matmul(rotation_matrix, pointcloud.T).T
            noise = np.random.normal(0, 0.02, pointcloud.shape)
            pointcloud += noise 
        
        pointcloud = torch.from_numpy(pointcloud)
        
        return {
            'pointcloud': pointcloud,
            'category': self.classes[category]
        }
    
    def __len__(self):
        return len(self.files)