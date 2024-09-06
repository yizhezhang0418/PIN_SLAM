import numpy as np
import torch
import open3d as o3d
import time
import os
import argparse
from tqdm import tqdm
from module import ply

class denoise_l0norm:
    def __init__(self, device='cuda') -> None:
        self.device = device
        self.pc_load = None
        self.origin_points = None
        self.normals = None
        self.kdtree = None

    def read_points(self, filepath):
        self.pc_load = o3d.io.read_point_cloud(filepath)
        self.origin_points = np.asarray(self.pc_load.points)
        return self.pc_load, self.origin_points

    def estimate_normals(self, k=20):
        if self.pc_load is None:
            raise ValueError("Point cloud is not loaded. Please load a point cloud first.")

        # Remove outliers
        k_out = 30
        radius_out = 3
        self.pc_load, ind = self.pc_load.remove_radius_outlier(nb_points=k_out, radius=radius_out)
        
        # Estimate normals
        self.pc_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

        # Convert points and normals to torch tensors and move to GPU
        points = torch.tensor(np.asarray(self.pc_load.points), device=self.device)
        normals = torch.tensor(np.asarray(self.pc_load.normals), device=self.device)

        # Build KDTree
        self.kdtree = o3d.geometry.KDTreeFlann(self.pc_load)

        # Calculate normalized normals
        norms = torch.norm(normals, dim=1, keepdim=True)
        normals /= norms

        n_hat = normals.clone()
        normals_temp = normals.clone()
        point_num, dimension = normals.shape

        diff_op = torch.zeros((k * point_num, dimension), device=self.device)
        sita = torch.ones((k * point_num, dimension), device=self.device, dtype=torch.float64)
        beta = 0.05
        nita = 0.10
        threshold = 0.1
        iternum = 0

        while torch.max(torch.abs(diff_op - sita)) > threshold:
            iternum += 1
            for i in range(point_num):
                # Perform KNN search using Open3D and convert indices to tensor
                _, idx, _ = self.kdtree.search_knn_vector_3d(self.pc_load.points[i], k)
                neighbor_indices = torch.tensor(idx, device=self.device, dtype=torch.long)
                
                neighbor_normals = normals_temp[neighbor_indices]  # Get neighbor normals in one go
                diffs = normals_temp[i] - neighbor_normals  # Calculate differences in one go
                diff_op[i*k:(i+1)*k] = diffs
                
                bb = torch.sum(diffs**2, dim=1)
                mask = (nita/beta > bb)
                sita[i*k:(i+1)*k][mask] = 0
                sita[i*k:(i+1)*k][~mask] = diffs[~mask]
                
                # Update normals
                normals[i] = (n_hat[i] + beta * torch.sum(neighbor_normals + sita[i*k:(i+1)*k], dim=0)) / (1 + beta * k)

            normals_temp = normals.clone()
            beta *= 2

        # Normalize the normals
        norms = torch.norm(normals, dim=1, keepdim=True)
        normals /= norms

        return normals.cpu().numpy()  # Move back to CPU and convert to numpy for saving

if __name__ == "__main__":
    output_folder = '/home/server/PIN_SLAM/data/wudasuidao/normal/40frame/ply_normalcorrect'
    directory_path = '/home/server/PIN_SLAM/data/wudasuidao/normal/40frame/ply'
    os.makedirs(output_folder, 0o755, exist_ok=True)

    for filename in tqdm(os.listdir(directory_path), desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            op = denoise_l0norm()
            op.pc_load, op.origin_points = op.read_points(filepath=file_path)
            op.normals = op.estimate_normals(k=20)
            op.pc_load.normals = o3d.utility.Vector3dVector(op.normals)

            op.pc_load.orient_normals_towards_camera_location()
            ply_file_path = os.path.join(output_folder, filename)

            field_names = ['x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z']
            array = np.hstack((op.pc_load.points, op.normals))
            if ply.write_ply(ply_file_path, [array[:, :6]], field_names):
                print("Export : " + ply_file_path)
            else:
                print('ply.write_ply() failed')
