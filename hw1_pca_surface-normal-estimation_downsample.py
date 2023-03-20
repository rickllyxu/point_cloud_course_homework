import enum
import glob
import logging
import os

import numpy as np
import open3d as o3d

logging.getLogger().setLevel(logging.INFO)

# data description:
# modelnet40_normal_resampled：其中txt文件中，一行有六个点：前三个点代表的是x,y,z。后三个点代表的是Nx，Ny，Nz，法向量。
data_root = r"C:\Users\kaiqiang.xu\Downloads\modelnet40_normal_resampled\modelnet40_normal_resampled"
shape_name_file = "modelnet40_shape_names.txt"
object_name = list(open(os.path.join(data_root, shape_name_file)).read().split('\n'))


####### Show data ###########
def visualize_data(window_name: str, points: np.array, normal_vector=None, colors=None):
    n, d = points.shape
    if d > 3:
        logging.warning("The point dimension: {}, expect dimension: 1, 2, or 3. "
                        "Abandon the redundant dimensions!".format(d))
        points = points[:][:3]
    if d == 1 or d == 2:
        logging.info("The point dimension: {}, would fill the missing dimensions".format(d))
        other_dimension = np.zeros(shape=(n, 3 - d))
        points = np.column_stack([points, other_dimension])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normal_vector is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal_vector)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920 // 2, height=1080 // 2, left=10, top=10, visible=True)
    vis.get_render_option().point_size = 2
    if normal_vector is not None:
        vis.get_render_option().point_show_normal = True
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


########### pca ###############
def normalize_by_center(object_data: np.array):
    center = object_data.mean(axis=0)
    assert center.shape[0] == object_data.shape[1]
    # object_data: n * d, center: d
    return object_data - center.reshape(1, -1)


def conduct_pca(object_data: np.array, target_dim=3) -> np.array:
    # original object_data: n * d
    # target object_data: n * target_dim
    object_data = normalize_by_center(object_data)
    u, _, _ = np.linalg.svd(np.matmul(object_data.transpose(), object_data))
    return np.matmul(object_data, u[:, :target_dim])


def question1_main():
    for obj in object_name[:3]:
        logging.info("Conduct pca on object type: {}".format(obj))
        obj_dir = os.path.join(data_root, obj)
        for file in os.listdir(obj_dir):
            file_path = os.path.join(obj_dir, file)
            logging.info(file_path)
            # x, y, z, nx, ny, nz
            data = np.genfromtxt(file_path, delimiter=',', dtype=float)[:, :3]
            visualize_data("original data: " + obj, data)
            reduced_data = conduct_pca(data, target_dim=2)
            visualize_data("pca data (downsample to 2 dim): " + obj, reduced_data)
            break


######### surface normal estimation for each point of each object ############


def surface_normal_estimation_per_point(data: np.ndarray):
    column_first_data = data.transpose()
    logging.info("Colum first data, shape: {}".format(column_first_data.shape))
    normal_vector = list()
    kd_tree = o3d.geometry.KDTreeFlann(column_first_data)
    knn = 100
    for point in data:
        k, idx, _ = kd_tree.search_knn_vector_3d(point.reshape(-1, 1), knn)
        # find the components
        u, s, _ = np.linalg.svd(column_first_data[:, idx])
        normal_vector.append(u[:, -1].reshape(-1))
    return np.array(normal_vector)


def question2_main():
    for obj in object_name[:3]:
        logging.info("Estimate surface normal of type: {}".format(obj))
        obj_dir = os.path.join(data_root, obj)
        for file in os.listdir(obj_dir):
            file_path = os.path.join(obj_dir, file)
            logging.info(file_path)
            # x, y, z, nx, ny, nz
            data = np.genfromtxt(file_path, delimiter=',', dtype=float)
            # visualize (nx, ny, nz)
            visualize_data("original points + original normal vector", points=data[:, :3], normal_vector=data[:, 3:])
            # compute nx_, ny_, nz_
            estimate_normal_vector = surface_normal_estimation_per_point(data[:, :3])
            visualize_data("original points + my estimate normal vector",
                           points=data[:, :3],
                           normal_vector=estimate_normal_vector)
            break


########### downsmaple each object using voxel grid downsampling ##########


def select_point_for_voxel_grid(voxel_grid_idx_list, data, select_method='exact'):
    selected_data = list()
    current_grid_idx = voxel_grid_idx_list[0][0]
    current_grid_point_end = 0
    while True:
        current_grid_point_start = current_grid_point_end
        while current_grid_point_end < len(voxel_grid_idx_list):
            if voxel_grid_idx_list[current_grid_point_end][0] == current_grid_idx:
                current_grid_point_end += 1
            else:
                current_grid_idx = voxel_grid_idx_list[current_grid_point_end][0]
                break
        if current_grid_point_end == current_grid_point_start:
            break
        if select_method == 'exact':
            points_idx = [u[1] for u in voxel_grid_idx_list[current_grid_point_start: current_grid_point_end]]
            points_in_grid = data[points_idx]
            selected_data.append(np.mean(points_in_grid, axis=0))

        elif select_method == 'random_select':
            gen_idx = np.random.randint(current_grid_point_start,
                                        current_grid_point_end)
            selected_data.append(data[voxel_grid_idx_list[gen_idx][1]])
    return np.array(selected_data)


def voxel_grid_downsampling(data):
    voxel_grid_size = 0.05

    logging.info("Compute each point's voxel grid index....")

    def compute_grid_num(one_dimension):
        min_, max_ = np.min(one_dimension), np.max(one_dimension)
        return min_, max_, (max_ - min_) / voxel_grid_size

    x_min, x_max, x_grid_num = compute_grid_num(data[:, 0])
    y_min, y_max, y_grid_num = compute_grid_num(data[:, 1])
    z_min, z_max, z_grid_num = compute_grid_num(data[:, 2])
    logging.info("x_grid_num: {}, y_grid_num: {}, z_grid_num: {}".format(x_grid_num,
                                                                         y_grid_num,
                                                                         z_grid_num))
    logging.info("The voxel grid num: {}".format(x_grid_num * y_grid_num * z_grid_num))

    def compute_point_grid_idx(point):
        x, y, z = point
        x_idx = np.ceil((x - x_min) / voxel_grid_size)
        y_idx = np.ceil((y - y_min) / voxel_grid_size)
        z_idx = np.ceil((z - z_min) / voxel_grid_size)
        return z_idx * x_grid_num * z_grid_num + y_idx * x_grid_num + x_idx

    voxel_grid_idx_list = list()
    for i, point in enumerate(data):
        voxel_grid_idx_list.append((compute_point_grid_idx(point), i))
    voxel_grid_idx_list.sort()

    logging.info("Select point for each voxel grid....")
    downsampled_data = select_point_for_voxel_grid(voxel_grid_idx_list, data, select_method='random_select')
    return downsampled_data


def question3_main():
    for obj in object_name[:3]:
        logging.info("Voxel grid downsampling object: {}".format(obj))
        obj_dir = os.path.join(data_root, obj)
        for file in os.listdir(obj_dir):
            file_path = os.path.join(obj_dir, file)
            logging.info(file_path)
            data = np.genfromtxt(file_path, delimiter=',', dtype=float)
            downsampled_data = voxel_grid_downsampling(data[:, :3])
            visualize_data("downsampled data", downsampled_data)


if __name__ == '__main__':
    # question1_main()
    # question2_main()
    question3_main()
