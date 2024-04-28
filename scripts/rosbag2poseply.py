import rosbag
import sensor_msgs.point_cloud2 as pc2

import os
import argparse
import numpy as np
import pandas as pd

from module import ply

import torch
from scipy.spatial.transform import Rotation

def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数转换为旋转矩阵。
    """
    # 四元数归一化
    q = quaternion / torch.norm(quaternion)
    qx, qy, qz, qw = q
    
    # 构造旋转矩阵
    R = torch.tensor([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def build_poses_from_df(df: pd.DataFrame, zero_origin=False):
    data = torch.from_numpy(df.to_numpy(dtype=np.float64))

    ts = data[:,0]
    xyz = data[:,1:4]
    quat = data[:,4:]

    rots = torch.from_numpy(Rotation.from_quat(quat).as_matrix())
    
    poses = torch.cat((rots, xyz.unsqueeze(2)), dim=2)

    homog = torch.Tensor([0,0,0,1]).tile((poses.shape[0], 1, 1)).to(poses.device)

    poses = torch.cat((poses, homog), dim=1)

    if zero_origin:
        rot_inv = poses[0,:3,:3].T
        t_inv = -rot_inv @ poses[0,:3,3]
        start_inv = torch.hstack((rot_inv, t_inv.reshape(-1, 1)))
        start_inv = torch.vstack((start_inv, torch.tensor([0,0,0,1.0], device=start_inv.device)))
        poses = start_inv.unsqueeze(0) @ poses

    return poses.float(), ts

def rosbag2ply(args, topics, ground_truth_file):

    os.makedirs(args.output_folder, 0o755, exist_ok=True)
    ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=",")
    # ground_truth_df = ground_truth_df.to_numpy(dtype=np.float64)
    begin_flag = False

    in_bag = rosbag.Bag(args.input_bag)
    count_for_pose=0
    for topic, msg, t in in_bag.read_messages(topics=topics):



        if not begin_flag:
            print(topic)





        if topic == args.topic:

            x,y,z,qx,qy,qz,qw=ground_truth_df.iloc[count_for_pose,:][1:]
            x = np.float64(x)
            y = np.float64(y)
            z = np.float64(z)
            qx = np.float64(qx)
            qy = np.float64(qy)
            qz = np.float64(qz)
            qw = np.float64(qw)
            count_for_pose=count_for_pose+1
            quat=torch.tensor([[qx],[qy],[qz],[qw]])
            rotation=quaternion_to_rotation_matrix(quat)
            gt_lidar_pose = torch.eye(4)
            translation=torch.tensor([[x], [y], [z]])
            gt_lidar_pose[:3,:3]=rotation
            gt_lidar_pose[:3,3]=translation.squeeze()

            gen = pc2.read_points(msg, skip_nans=True)
            data = list(gen)
            array = np.array(data)

            homogeneous_points = np.ones((array.shape[0], 4))
            homogeneous_points[:, :3] = array

            # 将 NumPy 数组转换为 Torch tensor
            homogeneous_points_torch = torch.tensor(homogeneous_points)

            # 应用变换矩阵
            transformed_points = torch.matmul(gt_lidar_pose.float(), homogeneous_points_torch.float().t())

            # 将结果转换回 NumPy 数组并且删除齐次坐标
            transformed_points_np = transformed_points.t()[:, :3].numpy()

            field_names = ['x','y','z']
            ply_file_path = os.path.join(args.output_folder, str(t)+".ply")

            if ply.write_ply(ply_file_path, [transformed_points_np[:,:3]], field_names):
                print("Export : "+ply_file_path)
            else:
                print('ply.write_ply() failed')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    parser.add_argument('-o','--output_folder', help="path for output foler")
    parser.add_argument('-t','--topic', help="name of the point cloud topic used in the rosbag", default="/cloud_block")
    args = parser.parse_args()
    print("usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to point cloud output folder] -t [name of point cloud rostopic]")
    
    topics=["/cloud_block", "/pose_block"]
    ground_truth_file = "./data/wudasuidao/one_frame/one_frame.csv"
    rosbag2ply(args, topics, ground_truth_file)
