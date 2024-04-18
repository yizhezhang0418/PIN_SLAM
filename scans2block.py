
import os
import sys
import rospy
import torch
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Path as Path_frame
from nav_msgs.msg import Odometry
import threading
import message_filters
import tf2_ros
from sensor_msgs.msg import PointCloud2
import numpy as np
# from pcl_ros import TransformPointCloud
from geometry_msgs.msg import TransformStamped
import tf_conversions
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Transform
from geometry_msgs.msg import PoseStamped
from queue import Queue

# PROJECT_ROOT = os.path.abspath(os.path.join(
#     os.path.dirname(__file__),
#     os.pardir))

# sys.path.append(PROJECT_ROOT)
# sys.path.append(PROJECT_ROOT + "/src")

# from examples.utils import *

class SyncedMessagesNode:
    """
    这个类最终的输出是blocks和blockpose

    返回:
    - self.blocks_laser:10个tensor点云,是列表形式
    - self.blockpose:位姿。
    - self.blockpose.header.pose.position.x是x坐标
    - self.blockpose.header.pose.orientation.x是旋转(四元数)
    """
    def __init__(self):
        rospy.init_node('synced_messages_node', anonymous=True)

        # 创建订阅者
        cloud_sub = message_filters.Subscriber('/cloud_registered_body', PointCloud2)
        odom_sub = message_filters.Subscriber('/Odometry', Odometry)
        path_sub = message_filters.Subscriber('/path', Path_frame)
        
        ########################## 创建发布者 ##############################
        self.block_laser_pub = rospy.Publisher('/cloud_block', PointCloud2, queue_size=10)
        self.block_path = rospy.Publisher('/pose_block',PoseStamped,queue_size=10)

        # 使用ApproximateTimeSynchronizer来同步消息
        self.ts = message_filters.ApproximateTimeSynchronizer([cloud_sub, odom_sub, path_sub], 50, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.sync_callback)

        # 初始化tf2的Buffer和Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.timestamp_blockcurrent=rospy.Time() # 这个block的第一帧的时间
        self.timestamp_blocklast=rospy.Time() # 这个block的最后一帧的时间
        self.timestamp_starttime=rospy.Time() # 这个系统的第一帧的时间
        self.is_first_frame=True
        # 初始化用于存储同步消息的变量
        self.packed_messages = []

        # 用于scan2block的数据集合 最终的结果
        self.blocks_laser = [] # 以list的形式，里面是torch的xyz点云
        self.blockpose = torch.eye(4) # 要转为torch类型4*4的变量

        # 这个block的队列
        self.block_laser_queue = Queue()
        self.block_pose_queue = Queue()
        self.timestamp_blockcurrent_queue=Queue()


        # 用于跳出is_shutdown的循环
        self.last_msg_time=rospy.Time.now()
        self.timeout = rospy.Duration(5)  # 设置5秒的超时
        rospy.loginfo("initialized scan2block")

        # thread = threading.Thread(target=lambda: rospy.spin())
        # thread.start()

    def sub_msg_inloop(self):
        cloud_sub = message_filters.Subscriber('/cloud_registered_body', PointCloud2)
        odom_sub = message_filters.Subscriber('/Odometry', Odometry)
        path_sub = message_filters.Subscriber('/path', Path_frame)
        # 使用ApproximateTimeSynchronizer来同步消息
        self.ts = message_filters.ApproximateTimeSynchronizer([cloud_sub, odom_sub, path_sub], 10000, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.sync_callback)

    def sync_callback(self, cloud_msg, odom_msg, path_msg):
        # 这是为了测试的
        # # 将同步的消息打包
        pc_header = cloud_msg.header
        odom_header = odom_msg.header
        path_header = path_msg.header
        # # 打印时间戳信息
        # rospy.loginfo(f"PointCloud2 Timestamp: {pc_header.stamp.to_sec()},Odometry Timestamp: {odom_header.stamp.to_sec()},path Timestamp: {path_header.stamp.to_sec()}")
        # rospy.loginfo(f"cloud_msg里面有的消息个数:{len(self.ts.queues[0])},odom_msg里面有的消息个数:{len(self.ts.queues[1])},path_msg里面有的消息个数:{len(self.ts.queues[2])}")
        
        # 记住第一帧的时间
        if(self.is_first_frame):
            rospy.sleep(0.01)
            self.timestamp_starttime=pc_header.stamp
            self.is_first_frame=False
            # rospy.loginfo(f"当前block第一帧的时间:{self.timestamp_starttime.to_sec()}")

        # 把最后一帧拿出来
        last_pose = path_msg.poses[-1] if path_msg.poses else None

        # 更新一下最后一帧的时间
        self.last_msg_time=rospy.Time.now()

        self.packed_messages.append((cloud_msg, odom_msg, last_pose)) # 在每一次使用前都记得清除，避免过大
        # print("checkpoint sync_callback")
        # 检查是否收集到足够的消息（例如10帧）
        if len(self.packed_messages) >= 10:
            self.blocks_laser,self.blockposemsg=self.scan2block() # 这个blocks就是可以用的点云block
            self.blockposemsg #这个时候是 self.blockposemsg.pose.position.(position or orientaion)
            # aa=self.blockposemsg.pose.position
            self.pose_quat=torch.tensor([[self.blockposemsg.pose.orientation.x],[self.blockposemsg.pose.orientation.y],
                                    [self.blockposemsg.pose.orientation.z],[self.blockposemsg.pose.orientation.w]])
            self.translation=torch.tensor([[self.blockposemsg.pose.position.x],[self.blockposemsg.pose.position.y],
                         [self.blockposemsg.pose.position.z]])
            rotation=self.quaternion_to_rotation_matrix(self.pose_quat)
            # 将旋转矩阵和平移向量放入变换矩阵
            self.blockpose[:3, :3] = rotation
            self.blockpose[:3, 3] = self.translation.squeeze() # 这里使用.squeeze()方法是为了去除translation向量中的任何单一维度，保证其可以正确地赋值给T矩阵
            
            self.timestamp_blocklast=pc_header.stamp # 当前block最后一帧stamp
            self.timestamp_blockcurrent=self.packed_messages[0][0].header.stamp # block的stamp 即block第一帧的stamp
            self.is_first_frame=True # 这里重置以下是否为第一帧
            
            ##############把laser和pose加入到queue里面##################
            self.block_laser_queue.put(self.blocks_laser)
            self.block_pose_queue.put(self.blockpose)
            self.timestamp_blockcurrent_queue.put(self.timestamp_blockcurrent)
            ##############subscribe每一个##################
            self.blocklaser2rosmsg()

            ##############清除每一个新的##################
            self.packed_messages.clear()



        # rospy.loginfo("Received a set of synchronized messages.")

    def blocklaser2rosmsg(self):
        msg = PointCloud2()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = "camera_init"
        points=self.blocks_laser.numpy()
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = False
        msg.data = np.asarray(points, np.float32).tobytes() # modified here yuanlaideshi tostring()

        # ros_msg = Path_frame()
        torch_pose=self.blockpose
        ros_pose = PoseStamped()


        # Extracting translation components
        ros_pose.pose.position.x = torch_pose[0, 3].item()
        ros_pose.pose.position.y = torch_pose[1, 3].item()
        ros_pose.pose.position.z = torch_pose[2, 3].item()

        # Extracting rotation components
        ros_pose.pose.orientation.x = float(self.pose_quat[0][0].item())
        ros_pose.pose.orientation.y = float(self.pose_quat[1][0].item())
        ros_pose.pose.orientation.z = float(self.pose_quat[2][0].item())
        ros_pose.pose.orientation.w = float(self.pose_quat[3][0].item())

        # Appending the pose to the path
        # ros_msg.poses.append(ros_pose)

        self.block_path.publish(ros_pose)
        self.block_laser_pub.publish(msg)
        print("published...")


    def blockpose2rosmsg(self,torch_pose):
        ros_msg = Transform()
        torch_pose=self.blockpose
        # Assuming torch_pose is a 4x4 tensor
        # Extract translation components
        ros_msg.translation.x = torch_pose[0, 3].item()
        ros_msg.translation.y = torch_pose[1, 3].item()
        ros_msg.translation.z = torch_pose[2, 3].item()

        ros_msg.rotation.x = self.pose_quat[0][0].item()
        ros_msg.rotation.y = self.pose_quat[1][0].item()
        ros_msg.rotation.z = self.pose_quat[2][0].item()
        ros_msg.rotation.w = self.pose_quat[3][0].item()

        return ros_msg

    def transform_point_cloud(point_cloud, transform):
        """
        将点云根据给定的变换矩阵进行变换。
        """
        transformed_cloud = []
        for point in point_cloud2.read_points(point_cloud, skip_nans=True):
            # 简化处理：假设点是(X, Y, Z)
            point_np = np.array([point[0], point[1], point[2], 1])
            transformed_point = np.dot(transform, point_np)
            transformed_cloud.append(transformed_point[:3])
        return transformed_cloud

             
    def calculate_transform(self, reference_pose_msg, current_pose_msg):
        """
        两帧之间的里程计
        """
        transform = TransformStamped()

        # 使用tf_conversions计算变换
        reference_pose = reference_pose_msg.pose
        current_pose = current_pose_msg.pose

        reference_matrix = tf_conversions.fromMsg(reference_pose)
        current_matrix = tf_conversions.fromMsg(current_pose)

        delta_matrix = reference_matrix.Inverse() * current_matrix
        transform.transform = tf_conversions.toMsg(delta_matrix)

        return transform

    def quaternion_to_rotation_matrix(self,quaternion):
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

    def apply_transform(self, cloud_msg, transform):
        """
        应用旋转和平移变换到点云。

        参数:
        xyz: torch.Tensor，形状为(N, 3)，表示点云的x, y, z坐标。
        translation: 一个长度为3的序列，表示x, y, z平移。
        quaternion: 一个长度为4的序列，表示旋转的四元数 (qx, qy, qz, qw)。

        返回:
        变换后的点云，为一个torch.Tensor。
        """
        ################ 首先把cloudmsg都转为xyz类型的点 ################
        points = point_cloud2.read_points_list(
                    cloud_msg, field_names=("x", "y", "z"))

        # xyz = torch.zeros((num_points, 3,), dtype=torch.float32)
        coords = [(p.x, p.y, p.z) for p in points]

        # 然后，将coords列表转换为NumPy数组
        coords_array = torch.tensor(coords, dtype=torch.float32)
        
        # 分离x, y, z坐标到不同的列中
        xyz = torch.zeros((len(points), 3), dtype=torch.float32)
        xyz[:, 0] = coords_array[:, 0]  # x坐标
        xyz[:, 1] = coords_array[:, 1]  # y坐标
        xyz[:, 2] = coords_array[:, 2]  # z坐标

        ################ 把变换拆解为四元数q和平移T ################
        translation=[transform.transform.position.x,transform.transform.position.y,
                     transform.transform.position.z]
        quaternion=[transform.transform.orientation.x,transform.transform.orientation.y,
                    transform.transform.orientation.z,transform.transform.orientation.w]

        # 将四元数转换为旋转矩阵
        translation = torch.tensor(translation, dtype=torch.float32)
        quaternion = torch.tensor(quaternion, dtype=torch.float32)
        rotate_mat=self.quaternion_to_rotation_matrix(quaternion)

        # 将点云旋转
        rotated_point_cloud = torch.matmul(xyz, rotate_mat.T)

        # 应用平移
        rotated_point_cloud = rotated_point_cloud + translation

        return rotated_point_cloud

    def scan2block(self):
        """
        把block的msg转为
        第一帧位姿+所有lidar点
        的形式
        返回:转换后的点云(torch)+第一帧的pose(msg形式)
        """ 
        # reference_odom = self.packed_messages[0][1]

        # 存储变换后的点云
        transformed_clouds = []

        # 提取并处理最近收集的10帧数据
        block = self.packed_messages[-10:]
        # 获取第一帧的位姿作为参考
        reference_pose=block[0][2]
        # 对每一帧进行转换
        for cloud_msg, _, pose_msg in block:
            # 计算当前帧相对于第一帧的变换
            transform = self.calculate_transform(reference_pose, pose_msg)

            ### msg 2 point torch.xyz and transform
            transformed_cloud=self.apply_transform(cloud_msg, transform)

            ### append torch
            transformed_clouds.append(transformed_cloud)
 
        # 拼接到一个点云变量里去
        merged_cloud = torch.cat(transformed_clouds, dim=0)
        rospy.loginfo(f"Processed 10 frames into a block.")
        return merged_cloud,reference_pose
   
    def gettime(self):
        return rospy.Time.now()

    def getdata(self):
        blockpose = self.blockpose
        blocks_laser = self.blocks_laser
        return blockpose, blocks_laser


if __name__ == '__main__':
    node_for_test=SyncedMessagesNode()
    # timer=rospy.Timer(rospy.Duration(2),SyncedMessagesNode().sync_callback)
    # rospy.spin()

    # thread0 = threading.Thread(target=SyncedMessagesNode())
    # thread0.start()

    thread1 = threading.Thread(target=lambda: rospy.spin())
    thread1.start()


    rospy.sleep(3)
    aa=11   
    while not rospy.is_shutdown():

        rospy.sleep(3)
        aa=11   



    # while not rospy.is_shutdown():
    #     timer=rospy.Timer(rospy.Duration(2),node_for_test.sync_callback)
    #     node_for_test





