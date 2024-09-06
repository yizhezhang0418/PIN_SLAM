#!/usr/bin/env python3
# @file      data_sampler.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import torch

from utils.config import Config
from utils.tools import get_time

class DataSampler():

    def __init__(self, config: Config):

        self.config = config
        self.dev = config.device


    # input and output are all torch tensors
    def sample(self, points_torch, 
               normal_torch,
               sem_label_torch,
               color_torch,
               normal_guided_sampling = True):
        # points_torch is in the sensor's local coordinate system, not yet transformed to the global system

        # T0 = get_time()

        dev = self.dev # DEVICE 0，GPU的编号，不用管
        surface_sample_range = self.config.surface_sample_range_m # noise level (actually as the std for a gaussian distribution) 
        surface_sample_n = self.config.surface_sample_n # =3
        freespace_behind_sample_n = self.config.free_behind_n # =1
        freespace_front_sample_n = self.config.free_front_n # =3
        # all_sample_n = surface_sample_n+freespace_behind_sample_n+freespace_front_sample_n+1 # 1 as the exact measurement
        all_sample_n = surface_sample_n + freespace_front_sample_n + 1 # 法向量-guided的采样 + SDF场之前的点 + 1(测量点云)
        free_front_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist = self.config.free_sample_end_dist_m
        sigma_base = self.config.sigma_sigmoid_m
        

        # get sample points points_torch(N,M),N为点的个数，M为点的其他属性(x,y,z)这类的
        point_num = points_torch.shape[0] # torch.shape[0]是原始神经点的个数
        distances = torch.linalg.norm(points_torch, dim=1, keepdim=True) # ray distances (scaled) 计算与原点的距离(points_torch is in local frame)
        
        """
            code for sample points
        """
        
        #################################### 新的采样的代码 ####################################
        # Part 0. the exact measured point # 这两个存的是采样点的东西
        measured_sample_displacement = torch.zeros_like(distances) # 这个是采样点的偏移，但是因为是实测点，所以为0 # [6122,1]
        measured_sample_dist_ratio = torch.ones_like(distances) # 这个是采样点与实测距离距离比值，因为是实测点，所以为1


        ############# Part 1. close-to-surface sampling 在法线上的采样点 #############
        # uniform sample in the close-to-surface range (+- range) (-1,1)
        surface_sample_ratio_uniform = (torch.rand(point_num*surface_sample_n, 1, device=dev)-0.5)*2 # 生成一个区间为[-1,1]的随机数
        
        # gaussian sampling (gaussian should provide both near surface samples and free space samples)
        self.config.gaussian_sampling_on = True
        if self.config.gaussian_sampling_on:
            # 生成一个均值为0，标准差为0.3的正态分布。这个0.3应该是为了3sigma原则设定的。
            surface_sample_ratio_gaussian = torch.randn(point_num*surface_sample_n,1,device=dev)*0.3 # [M,1]
            condition = torch.logical_and(surface_sample_ratio_gaussian > -1, surface_sample_ratio_gaussian < 1) # 在[-1,1]这个区间
            # torch.where(condition, a, b) 函数用于基于条件张量 condition 的值来选择 a 和 b 之间的元素。
            # 如果 condition 中的元素为 True，则相应位置的元素从 surface_sample_ratio_gaussian 中取；如果为 False，则从 surface_sample_ratio_uniform 中取。
            surface_sample_ratio = torch.where(condition, surface_sample_ratio_gaussian, surface_sample_ratio_uniform) # condition true就选前面，否则后面
            #print(surface_sample_ratio)
        else:
            surface_sample_ratio = surface_sample_ratio_uniform
        # 以上的代码都是随机生成采样点的过程

        # # 生成完之后乘一个比例，无量纲转到有量纲
        surface_sample_displacement = surface_sample_ratio * surface_sample_range# 在表面周围采样的大小(m) surface_sample_ratio * 0.25
        # surface_sample_displacement = torch.randn(point_num*surface_sample_n, 1, device=dev)*surface_sample_range # new
        
        repeated_dist = distances.repeat(surface_sample_n,1) # 距离矩阵(M,1)，M代表实际测量到的点的个数。重复3次，(3M,1).相当于是有M条射线，在每个射线上采样3个点
        # 下面这句话的意思其实就是你采样的点相较于光线dist的比值
        surface_sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface (3M,1)

        surface_repeated_points = points_torch.repeat(surface_sample_n,1) # 实测点重复[3M,3]
        if normal_guided_sampling:
            normal_direction = normal_torch.repeat(surface_sample_n,1) # normals are oriented towards sensors.[3M,3]
            #note that normals are oriented towards origin (inwards) 法线朝向原点
            # 用法线改正的采样点(local frame下)得到了采样点在local frame下面的采样点的坐标
            surface_sample_points = surface_repeated_points + surface_sample_displacement * (-normal_direction) # [3M,3] + [3M,1] * [3M,3]
        else:
            # 没有用法线改正的采样点，这些采样点沿着光线
            surface_sample_points = surface_repeated_points * surface_sample_dist_ratio

        # only near surface samples are assigned to semantic labels.
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(1, surface_sample_n).transpose(0,1)
        
        ############# Part 2. free space uniform sampling ############# 这个采样是沿着ray方向进行采样的东西
        repeated_dist = distances.repeat(freespace_front_sample_n,1)
        if sem_label_torch is not None:
            free_sem_label_tensor = torch.zeros_like(repeated_dist)

        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0 # free_sample_end_dist = 0.6  1.0几一点点
        free_diff_ratio = free_max_ratio - free_front_min_ratio # free_front_min_ratio = 0.3
        free_sample_dist_ratio = torch.rand(point_num*freespace_front_sample_n, 1, device=dev)*free_diff_ratio + free_front_min_ratio
        free_sample_displacement = (free_sample_dist_ratio - 1.0) * repeated_dist
        free_repeated_points = points_torch.repeat(freespace_front_sample_n,1)
        free_sample_points = free_repeated_points * free_sample_dist_ratio

        # remove near-surface samples from free-space samples
        tr = surface_sample_range*1.33
        valid_mask = torch.logical_or(free_sample_displacement < -tr, free_sample_displacement > tr).reshape(-1) # 筛选掉tr范围之外的采样点
        free_sample_displacement = free_sample_displacement[valid_mask]
        free_sample_points = free_sample_points[valid_mask]

        free_sample_displacement
        free_sample_points
        
        """
            在这里之后的代码就是采样完点的代码
        """
        
        # all together
        all_sample_points = torch.cat((points_torch, 
                                       surface_sample_points,
                                       free_sample_points),0) # 这个是坐标系下的点 [40769,3]
        # measured_sample_displacement代表实测点
        all_sample_displacement = torch.cat((measured_sample_displacement, 
                                             surface_sample_displacement, 
                                             free_sample_displacement),0) # [40769,1]=[6122,1],[18366,1],[16281,1]
        # 采样点按照：1.实际点；2.法线采样点；3.ray方向的采样点

        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(all_sample_displacement) # [40769,1]

        surface_sample_count = point_num * (surface_sample_n + 1) # 包含了实测点+法线采样点
        self.config.dist_weight_on = False
        if self.config.dist_weight_on: # far away surface samples would have lower weight
            weight_tensor[:surface_sample_count] = 1 + self.config.dist_weight_scale * 0.5 - \
                (repeated_dist[:surface_sample_count] / self.config.max_range) * self.config.dist_weight_scale # [0.6, 1.4]
        # TODO: also add lower weight for surface samples with large incidence angle

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if self.config.behind_dropoff_on:
            dropoff_min = 0.2 * free_sample_end_dist
            dropoff_max = free_sample_end_dist
            dropoff_diff = dropoff_max - dropoff_min
            # behind_displacement = (repeated_dist*(all_sample_dist_ratio-1.0)/sigma_base)
            behind_displacement = all_sample_displacement
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min = 0.0, max = 1.0)
            dropoff_weight = dropoff_weight * 0.8 + 0.2
            weight_tensor = weight_tensor * dropoff_weight
        
        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        # weight_tensor[surface_sample_count:] *= -1.0 
        weight_tensor[point_num*(surface_sample_n+1):] *= -1.0 
        
        # ray-wise depth
        # distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front - 
        # 修改了这个下面这个式子的符号by alex 2024.5.29
        sdf_label_tensor = - all_sample_displacement.squeeze(1)  # scaled [-1, 1] # as distance (before sigmoid)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            surface_normal = normal_torch.repeat(surface_sample_n,1)
            free_normal = normal_torch.repeat(freespace_front_sample_n,1)
            free_normal = free_normal[valid_mask]
            normal_label_tensor = torch.cat((normal_torch, surface_normal, free_normal),0) # [40769,3]
            # normal_label_tensor = normal_torch.repeat(all_sample_n, 1)
        
        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None

        # assign the color label to the close-to-surface samples
        color_tensor = None

        all_sample_points = all_sample_points.reshape(-1, 3)
        sdf_label_tensor = sdf_label_tensor.reshape(-1) 

        weight_tensor = weight_tensor.reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = normal_label_tensor.reshape(-1, 3)

        # print("time for sampling I:", T1-T0)
        # print("time for sampling II:", T2-T1)
        # print("time for sampling III:", T3-T2)
        # all super fast, all together in 0.5 ms

        return all_sample_points, sdf_label_tensor, normal_label_tensor, sem_label_tensor, color_tensor, weight_tensor

    
    def sample_source_pc(self, points):

        dev = self.dev
        sample_count_per_point = 0 
        sampel_max_range = 0.2

        if sample_count_per_point == 0: # use only the original points
            return points, torch.zeros(points.shape[0], device=dev)
        
        unground_points = points[points[:,2]> -1.5]
        point_num = unground_points.shape[0]

        repeated_points = unground_points.repeat(sample_count_per_point,1)

        surface_sample_displacement = (torch.rand(point_num*sample_count_per_point, 1, device=dev)-0.5)*2*sampel_max_range 
        
        distances = torch.linalg.norm(unground_points, dim=1, keepdim=True) # ray distances 

        repeated_dist = distances.repeat(sample_count_per_point,1)
        sample_dist_ratio = surface_sample_displacement/repeated_dist + 1.0 # 1.0 means on the surface

        sample_points = repeated_points*sample_dist_ratio
        sample_labels = -surface_sample_displacement.squeeze(-1)

        sample_points = torch.cat((points, sample_points), 0)
        sample_labels = torch.cat((torch.zeros(points.shape[0], device=dev), sample_labels), 0)

        return sample_points, sample_labels