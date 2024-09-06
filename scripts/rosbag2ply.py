import rosbag
import sensor_msgs.point_cloud2 as pc2

import os
import argparse
import numpy as np

from module import ply

# import open3d as o3d

def rosbag2ply(args):

    os.makedirs(args.output_folder, 0o755, exist_ok=True)
    
    begin_flag = False

    in_bag = rosbag.Bag(args.input_bag)
    for topic, msg, t in in_bag.read_messages():

        if not begin_flag:
            print(topic)

        if topic == args.topic:
            gen = pc2.read_points(msg, skip_nans=True)
            data = list(gen)
            array = np.array(data)

            # NOTE: point cloud array: x,y,z,intensity,timestamp,ring,others...
            # could be different for some other rosbags
            # print(array[:, :6])
            
            # timestamps = array[:, 4] # for hilti and others
            # timestamps = array[:, 5] # for m2dgr
            # print(timestamps)

            # if not begin_flag:
            #     shift_timestamp = timestamps[0]
            #     begin_flag = True

            # timestamps_shifted = timestamps - shift_timestamp
            # print(timestamps_shifted)

            ## 原来的
            field_names = ['x','y','z','normal_x','normal_y','normal_z']
            ply_file_path = os.path.join(args.output_folder, str(t)+".ply")

            if ply.write_ply(ply_file_path, [array[:,:6]], field_names):
                print("Export : "+ply_file_path)
            else:
                print('ply.write_ply() failed')


            # points = array[:, :3]  # 提取 x, y, z
            # normals = array[:, 3:6]  # 提取 normal_x, normal_y, normal_z

            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            # point_cloud.normals = o3d.utility.Vector3dVector(normals)

            # ply_file_path = os.path.join(args.output_folder, str(t)+".ply")
            # o3d.io.write_point_cloud(ply_file_path, point_cloud, write_ascii=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    parser.add_argument('-o','--output_folder', help="path for output foler")
    parser.add_argument('-t','--topic', help="name of the point cloud topic used in the rosbag", default="/cloud_block")
    args = parser.parse_args()
    print("usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to point cloud output folder] -t [name of point cloud rostopic]")
    
    rosbag2ply(args)
