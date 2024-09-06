import numpy as np
import wandb
import torch
import pcl
import open3d as o3d
import time
import os
import argparse
from tqdm import tqdm
from module import ply

class denoise_l0norm:
    def __init__(self) -> None:
        self.pc_load = None
        self.origin_points = None
        self.smooth_points = None
        # self.neighbors_normal_index={}
        self.normals = None
        self.nita = 0
        self.sita = 0
        self.beta = 0


    def read_points(self, filepath):

        self.pc_load = o3d.io.read_point_cloud(filepath)
        # self.origin_points = np.asarray(self.pc_load.points)
        return self.pc_load, self.origin_points

    def estimate_normals(self, k = 20):
        if self.pc_load is None:
            raise ValueError("Point cloud is not loaded. Please load a point cloud first.")
        
        ## step1: Use PCA for Normal Estimation
        # 剔除近邻点少的点，相当于作了一次radius剔除
        # parameters
        # nb_points：选择球体中最少点的数量。 radius：用来计算点的邻域点的数量的球的半径。
        k_out = 15
        radius_out = 3
        self.pc_load, ind = self.pc_load.remove_radius_outlier(nb_points=k_out, radius=radius_out) # cl是pointcloud;ind是剩下的点的索引

        # knn estimate
        k = 25
        radius=0.3
        self.pc_load.estimate_normals(
            # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.3, max_nn = k)
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn = k)
        )
        # knn后调整法线
        # self.pc_load.orient_normals_towards_camera_location()
        self.pc_load.orient_normals_consistent_tangent_plane(100)
        # o3d.visualization.draw_geometries([self.pc_load],
        #                         # zoom=0.3412,
        #                         # front=[0.4257, -0.2125, -0.8795],
        #                         # lookat=[2.6172, 2.0475, 1.532],
        #                         # up=[-0.0694, -0.9768, 0.2024],
        #                         point_show_normal=True)
        
        neighbors_normal_index={}

        ## neighbors
        self.kdtree = o3d.geometry.KDTreeFlann(self.pc_load)

        T1 = time.time()

        self.origin_points = np.asarray(self.pc_load.points)
        # Only costs 1.11s-> 0.3s
        for i in range(len(self.origin_points)):
            [a,idx,b] = self.kdtree.search_knn_vector_3d(self.pc_load.points[i], k)
            neighbors_normal_index[i] = idx

        # kk = self.neighbors[1][0]

        T2 = time.time()
        # print(f"Time cost:{T2-T1}")

        normals = np.asarray(self.pc_load.normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # 归一化
        normals = normals / norms
        # # 可选：根据视点位置调整法线方向
        # self.pc_load.orient_normals_towards_camera_location()


        n_hat = np.copy(normals)
        normals_temp = np.copy(normals)
        [point_num, dimension] = normals.shape

        diff_op = np.zeros((k * point_num, dimension))
        sita = np.ones((k * point_num, dimension))
        beta = 0.05 # 0.01
        nita = 0.10 # 0.05
        threshold = 0.1
        iternum = 0

        while np.max(np.abs(diff_op - sita)) > threshold:
            iternum += 1
            for i in range(point_num):
                neighbor_indices = np.array(neighbors_normal_index[i])  # 获取第i个点的邻居索引
                neighbor_normals = normals_temp[neighbor_indices]  # 直接获取所有邻居的法向量
                diffs = normals_temp[i] - neighbor_normals  # 一次计算所有差值
                diff_op[i*k:(i+1)*k] = diffs
                
                bb = np.sum(diffs**2, axis=1)
                mask = (nita/beta > bb)
                sita[i*k:(i+1)*k][mask] = 0
                sita[i*k:(i+1)*k][~mask] = diffs[~mask]
                
                # 计算新的 normals[i] 值
                normals[i] = (n_hat[i] + beta * np.sum(neighbor_normals + sita[i*k:(i+1)*k], axis=0)) / (1 + beta * k)
            
            normals_temp = np.copy(normals)


            # print(f"The iteration number is {iternum}, the biggest diff is {np.max(np.abs(diff_op - sita))}, beta is {beta}")
            beta *= 2

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= norms  # 归一化

        # self.pc_load.orient_normals_towards_camera_location()
        # self.pc_load.

        # -normals是因为要使得法向量朝内
        return -normals

    def point_denoise(self, k, normals):
        origin_points = self.origin_points
        p_hat = np.copy(origin_points)
        points_iter = np.copy(origin_points)
        points_temp = np.copy(origin_points)
        normals # 优化后法向量
        # k = 10

        neighbors_point={}
        for i in range(len(origin_points)):
            [_,idx,_] = self.kdtree.search_knn_vector_3d(self.pc_load.points[i], k)
            neighbors_point[i] = idx

        [point_num, dimesion] = origin_points.shape
        # diff_op_point = np.zeros((k*point_num, dimesion))
        kesai = np.ones((k*point_num, 1)) # 辅助变量
        DP_ikj = np.zeros((k*point_num, 1)) # 新的DP
        DP_ikj_pre = np.zeros((k*point_num, 1)) # 上一次迭代的，即DP～
        alpha = np.eye(point_num, dtype=float) # alpha对角阵
        delta = 0.005 # 0.002-0.008
        t = 0.008 # 辅助变量系数
        threshold = 0.01
        iternum = 0

        while(np.abs(np.max(DP_ikj - kesai)) > threshold):
            iternum = iternum + 1
            ## 第一个优化的解
            for i in range(point_num):
                DP_ikj_pre = np.copy(DP_ikj) #存储上一次的
                for j in range(k):
                    # diff_op_point[i*k+j,:] = (points_iter[i,:] - points_iter[neighbors_point[i][j],:])*normals[i,:]

                    # 这一次的Dp
                    DP_ikj[i*k+j] = np.dot(normals[i,:] ,(points_iter[i,:] - points_iter[neighbors_point[i][j],:]))
                    if (t/delta > np.square(DP_ikj[i*k+j])):
                        kesai[i*k+j] = 0
                    else:
                        kesai[i*k+j] = np.copy(DP_ikj[i*k+j])
                
                # for j in range(k):
                #     alpha[i,i] = ((p_hat[i,:] + delta * ((points_iter[neighbors_point[i][j],:] + alpha[neighbors_point[i][j],neighbors_point[i][j]] * normals[neighbors_point[i][j],:])*normals[i,:] + kesai[i*k+j,:])) / (1 + delta * normals[i,:]) - points_iter[i,:]) / normals[i,:]

                # DP_ikj_pre = np.copy(DP_ikj) #存储上一次的

                for j in range(k):
                    term1 = np.dot((points_temp[i,:] - p_hat[i,:]) , normals[i,:])
                    term2 = delta * (DP_ikj_pre[i*k+j,:] - np.dot(normals[i,:],normals[neighbors_point[i][j],:])*alpha[neighbors_point[i][j],neighbors_point[i][j]] - kesai[i*k+j])
                    term3 = np.dot(normals[i,:], normals[i,:]) + delta
                    alpha[i,i] = -1 * (term1+term2) / term3
                
                points_iter[i,:] = points_temp[i,:] + alpha[i,i] * normals[i,:]
            points_temp = np.copy(points_iter)

            print(f"The iteration number is {iternum} , the biggest diff is {np.abs(np.max(DP_ikj - kesai))} , delta is {delta}")
            delta = 2 * delta

        
        return points_iter



if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    # parser.add_argument('-o','--output_folder', help="path for output foler")
    # args = parser.parse_args()
    output_folder = '/home/server/PIN_SLAM/data/wudasuidao/normal/30frame/ply_normalcorrect_dir_100_kout_15_knn_25'
    directory_path = '/home/server/PIN_SLAM/data/wudasuidao/normal/30frame/ply'
    os.makedirs(output_folder, 0o755, exist_ok=True)

    for filename in tqdm(os.listdir(directory_path), desc="Processing files"):
        file_path = os.path.join(directory_path, filename)
        
        # 检查是否为文件（而不是目录）
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:

                op = denoise_l0norm()
                # filepath = "/home/server/PIN_SLAM/test/tunnel.ply"
                op.pc_load, op.origin_points = op.read_points(filepath=file_path)
                op.normals = op.estimate_normals(k = 20)
                op.pc_load.normals = o3d.utility.Vector3dVector(op.normals)

                # op.pc_load.orient_normals_towards_camera_location()
                ply_file_path = os.path.join(output_folder, filename)

                field_names = ['x','y','z','normal_x','normal_y','normal_z']
                array = np.hstack((op.pc_load.points, op.normals))
                if ply.write_ply(ply_file_path, [array[:,:6]], field_names):
                    print("Export : "+ply_file_path)
                else:
                    print('ply.write_ply() failed')


            # o3d.visualization.draw_geometries([op.pc_load],
            #                                 # zoom=0.3412,
            #                                 # front=[0.4257, -0.2125, -0.8795],
            #                                 # lookat=[2.6172, 2.0475, 1.532],
            #                                 # up=[-0.0694, -0.9768, 0.2024],
            #                                 point_show_normal=False)