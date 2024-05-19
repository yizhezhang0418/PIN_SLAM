import open3d as o3d
import numpy as np

def point_cloud_to_mesh(point_cloud):
    # 将点云转换为网格
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud)
    return mesh

def main():
    # 读取点云文件
    point_cloud = o3d.io.read_point_cloud("005.ply")
    print("success load")

    # 计算点云的法线
    # point_cloud.estimate_normals()
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print("success estimate_normals")

    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud,o3d.utility.DoubleVector([radius, radius * 2]))

    bpa_mesh.remove_degenerate_triangles()
    bpa_mesh.remove_duplicated_triangles()
    bpa_mesh.remove_duplicated_vertices()
    bpa_mesh.remove_non_manifold_edges()

    # 使用泊松表面重建算法生成网格
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)[0]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9, width=0, scale=1.1, linear_fit=False)[0]

    # print("success TriangleMesh")
    # 可视化网格
    # o3d.visualization.draw_geometries([mesh])

    # 保存网格为 PLY 文件
    o3d.io.write_triangle_mesh("output_mesh.ply", bpa_mesh)

if __name__ == "__main__":
    main()
