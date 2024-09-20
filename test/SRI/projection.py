import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import open3d as o3d

import math
from scipy.ndimage import filters, convolve
from scipy import ndimage

def initialize_lidar(file_path):
    """
    Initialize a LiDAR having given laser resolutions from a configuration file.

    :param file_path: path of LiDAR configuration file [.yaml]
    :param channels: number of vertical angles (vertical resolution)
    :param points_per_ring: number of horizontal angles (horizontal resolution)
    :return: LiDAR specification
    """
    with open(file_path, 'r') as f:
        lidar = yaml.load(f, Loader=yaml.FullLoader)

    lidar['max_v'] *= (np.pi / 180.0)  # [rad]
    lidar['min_v'] *= (np.pi / 180.0)  # [rad]
    lidar['max_h'] *= (np.pi / 180.0)  # [rad]
    lidar['min_h'] *= (np.pi / 180.0)  # [rad]

    return lidar

def points_to_ranges(points):
    """
    Convert points in the sensor coordinate into the range data in spherical coordinate.

    :param points: points in sensor coordinate
    :return: the range data in spherical coordinate
    """
    # sensor coordinate
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x * x + y * y + z * z)
    v = np.arctan2(z, np.sqrt(x * x + y * y)) # vertical 
    h = np.arctan2(x, y) # horizon

    # r = np.sqrt(x * x + y * y + z * z)
    # h = np.arctan2(x, z) # horizon, sita
    # v = np.arcsin(y, np.sqrt(x * x + y * y + z * z)) # vertical, fai

    return np.stack((v, h, r), axis=-1)


def store_transformation_mat(lidar):
    # range_samples = points_to_ranges(points) # return spherical coordinate [r,v,h]
    
    max_y = lidar['max_v'] + 10 * np.pi / 180
    min_y = lidar['min_v'] - 10 * np.pi / 180
    max_x = lidar['max_h'] + 10 * np.pi / 180
    min_x = lidar['min_h'] - 10 * np.pi / 180
    res_y = 0.28 * np.pi / 180
    res_x = 0.03 * np.pi / 180

    # 计算图像矩阵的xy坐标
    num_x = math.ceil((max_x - min_x) / res_x)
    num_y = math.ceil((max_y - min_y) / res_y)

    # 创建一个存储转换矩阵的数组
    transformation_matrices = np.zeros((num_y, num_x, 3, 3), dtype=np.float32)

    for i in range(num_y):
        for j in range(num_x):
            fai = min_x + j * res_x
            theta = min_y + i * res_y

            T = np.array([[-np.sin(fai), np.cos(fai)*np.cos(theta), np.cos(fai)*np.sin(theta)],
                          [ np.cos(fai), np.sin(fai)*np.cos(theta), np.sin(fai)*np.sin(theta)],
                          [ 0.0        , -np.sin(theta)           , np.cos(theta)            ]])
            
            transformation_matrices[i,j] = T
    
    return transformation_matrices




def points_to_range_image(points, lidar):
    """
    Convert points in the sensor coordinate to a range image.

    :param points: points in sensor coordinate
    :param lidar: LiDAR specification
    :return: range image of which a value has the range [0 ~ norm_r]
    """
    range_samples = points_to_ranges(points) # return spherical coordinate [v, h, r]

    # 0.28° (竖直) × 0.03° (水平)
    # range_image = np.zeros([lidar['channels'], lidar['points_per_ring']], dtype=np.float32)
    #            0.6474657443845809,           -0.6546454752730432,          0.6317975058155382,           -0.5922645856211209
    # print(f'{np.max(range_samples[:, 0])},{np.min(range_samples[:, 0])},{np.max(range_samples[:, 1])},{np.min(range_samples[:, 1])}')
    max_y = lidar['max_v']
    min_y = lidar['min_v']
    max_x = lidar['max_h']
    min_x = lidar['min_h']
    res_y = 0.28 * np.pi / 180
    res_x = 0.03 * np.pi / 180
    range_image = np.zeros([math.ceil((max_y-min_y)/res_y) + 1, math.ceil((max_x-min_x)/res_x) + 1], dtype=np.float32)

    # prune outliers
    # 使用布尔索引筛选符合条件的点 [v, h, r]

    mask = ((range_samples[:, 0] >= min_y) & (range_samples[:, 0] <= max_y) &
            (range_samples[:, 1] >= min_x) & (range_samples[:, 1] <= max_x))
    filtered_samples = range_samples[mask]
    filtered_points = points[mask]
    # # 打印结果查看
    # print(f'Filtered array shape: {filtered_samples.shape}')

    # # 如果需要查看筛选后的最大值和最小值
    print(f'{np.max(filtered_samples[:, 0])},{np.min(filtered_samples[:, 0])},{np.max(filtered_samples[:, 1])},{np.min(filtered_samples[:, 1])}')


    ### test for transformation_matrices
    transformation_matrices = np.zeros((math.ceil((max_y-min_y)/res_y) + 1, math.ceil((max_x-min_x)/res_x) + 1, 3, 3), dtype=np.float32)




    # offset to match a point into a pixel center
    filtered_samples[:, 0] += (res_y * 0.5) # 可能会有超出界限的，需要+1
    filtered_samples[:, 1] += (res_x * 0.5)
    # horizontal values are within [-pi, pi) 
    filtered_samples[filtered_samples[:, 1] < -np.pi, 1] += (2.0 * np.pi)
    filtered_samples[filtered_samples[:, 1] >= np.pi, 1] -= (2.0 * np.pi)

    # py,px对应着points的点的序号                                                                            别卷了  放假了 0.0TAT 实验做不出来只能苦碧加班 会不会打字 这个ubuntu输入法台垃圾了 pyn问你在哪 快来实验室玩 来不了一点，只能等明天上课再来，，，狠狠陪npy是把 哈哈哈哈哈哈哈哈 在家那是 必须的   建议把这一段注释保留，作为你论文代码的彩蛋 论文发不出去了 各位大佬捞捞我
    py = np.trunc((filtered_samples[:, 0] - min_y) / res_y).astype(int) # start from 0
    px = np.trunc((filtered_samples[:, 1] - min_x) / res_x).astype(int) # mid70和Velodyne不一样，不能用pi去减
    # print(px.max())
    # print(py.max())
    range_image[py, px] = filtered_samples[:, 2] # [265,2348]

    """
        使用高斯滤波预处理和 Prewitt 算子计算深度图的法向量。

        参数:
        depth_map: 2D NumPy 数组，表示深度图。

        返回:
        normals: 3D NumPy 数组，形状为 (H, W, 3)，每个位置存储一个法向量。
    """
    # Step 1: 高斯预滤波
    # 使用 sigma=1 的 3x3 高斯滤波器
    filtered_depth_map = filters.gaussian_filter(range_image, sigma=1)
    # Step 2: Prewitt 算子计算梯度
    # # 定义 Prewitt 算子的核
    # prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # x方向
    # prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # y方向

    # # 应用 Prewitt 算子
    # dz_dx = convolve(filtered_depth_map, prewitt_kernel_x)  # x 方向梯度
    # dz_dy = convolve(filtered_depth_map, prewitt_kernel_y)  # y 方向梯度



    # 使用Prewitt算子进行边缘检测
    prewitt_x = ndimage.prewitt(filtered_depth_map, axis=0)  # 水平方向
    prewitt_y = ndimage.prewitt(filtered_depth_map, axis=1)  # 垂直方向


    # Step 3: 计算法向量
    # H, W = range_image.shape
    normals = np.zeros_like(filtered_samples) # [96256,3]
    # normals = np.zeros((H, W, 3), dtype=np.float32)

    # 改一个for，循环赋值（but time-costing?)
    normals[:, 0] = prewitt_x[py, px]
    normals[:, 1] = prewitt_y[py, px]
    normals[:, 2] = 1
    # normals_product = prewitt_x * prewitt_y
    # normals = normals_product[py, px]

    # 对法向量进行归一化处理
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norm


    for i in range(math.ceil((max_y-min_y)/res_y) + 1):
        for j in range(math.ceil((max_x-min_x)/res_x) + 1):
            fai = min_x + j * res_x
            theta = min_y + i * (res_y)
            # T = np.array([[np.cos(fai),                -np.sin(fai),               0.0],
            #               [np.cos(theta)*np.sin(fai),  np.cos(theta)*np.cos(fai),  -np.sin(theta)],
            #               [np.sin(theta)*np.sin(fai),  np.sin(theta)*np.cos(fai),  np.cos(theta)]])
            T = np.array([[np.sin(fai),                -np.cos(fai),               0.0],
                          [np.cos(theta) * np.cos(fai), np.cos(theta) * np.sin(fai), -np.sin(theta)],
                          [np.sin(theta) * np.cos(fai), np.sin(theta) * np.sin(fai),  np.cos(theta)] ])
            # T = np.array([[np.sin(theta)*np.cos(fai), np.cos(theta),  -np.sin(theta)*np.sin(fai)],
            #               [np.sin(fai),               0.0,            np.cos(theta)],
            #               [np.cos(fai)*np.cos(theta), -np.sin(theta), -np.cos(theta)*np.sin(fai)]])

            transformation_matrices[i,j] = T

    selected_tf_mat = transformation_matrices[py, px]

    normals_cartesian = np.einsum('nij,nj->ni', selected_tf_mat, normals)


    return range_image, normals_cartesian, filtered_points


def range_image_estimate_normals(range_im, points):
    normals = np.zeros_like(points)
    sita = np.pi / 2 - ()

    return normals

if __name__ == '__main__':
    # init lidar
    lidar_path = '/home/server/PIN_SLAM/test/SRI/lidar_specification.yaml'
    lidar = initialize_lidar(lidar_path)
    
    # read points
    # filepath = "/home/server/PIN_SLAM/data/wudasuidao/test_normal_extraction/1.ply"
    filepath = "/home/server/PIN_SLAM/data/wudasuidao/normal/30frame/1.ply"
    pc_load = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pc_load.points) # 点云

    # pc_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 20))

    # pc_load.orient_normals_consistent_tangent_plane(100)
    
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pc_load, coordinate_frame],
    #                             # zoom=0.3412,
    #                             # front=[0.4257, -0.2125, -0.8795],
    #                             # lookat=[2.6172, 2.0475, 1.532],
    #                             # up=[-0.0694, -0.9768, 0.2024],
    #                             point_show_normal=True
    #                             )
    
    
    # tf_mat = store_transformation_mat(lidar=lidar)
    normals = np.zeros_like(points)
    # convert to range view
    range_im, normals, points_filter= points_to_range_image(points, lidar)

    pc_load.points = o3d.utility.Vector3dVector(points_filter)
    pc_load.normals = o3d.utility.Vector3dVector(normals)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc_load, coordinate_frame],
                                # zoom=0.3412,
                                # front=[0.4257, -0.2125, -0.8795],
                                # lookat=[2.6172, 2.0475, 1.532],
                                # up=[-0.0694, -0.9768, 0.2024],
                                point_show_normal=True
                                )

    # plt.imshow(range_im)
    plt.imsave('range_im.png', range_im)
    # plt.colorbar()  # 显示颜色条
    # plt.title('Random Image')  # 设置标题
    # plt.show()

