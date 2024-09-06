import pcl
import numpy as np

# 读取点云
cloud = pcl.load('/home/server/PIN_SLAM/test/bun0.pcd')

# 创建MLS对象并设置参数
mls = cloud.make_moving_least_squares()
mls.set_search_radius(0.03)
mls.set_polynomial_fit(True)
mls.set_polynomial_order(2)
# mls.set_up_sample_method(pcl.MovingLeastSquares.SAMPLE_LOCAL_PLANE)

# 执行MLS
mls.process()

# 获取处理后的点云及其法向量
mls_points = mls.get_output()
normals = mls.get_normals()

# 保存处理后的点云及其法向量
pcl.save(mls_points, 'mls_output_cloud.pcd')
np.savetxt('mls_normals.txt', normals, delimiter=' ')

print("MLS处理完成并保存点云及其法向量。")
