import numpy as np
import trimesh


class CurvePathPlanner:
    """
    曲面路径规划算法库
    包含：动态步长螺旋、Zigzag、路径分段重连、ROI过滤
    参数均可在GUI界面修改
    """

    @staticmethod
    def ensure_single_mesh(mesh_or_list):
        # 确保输入是单个trimesh对象
        if isinstance(mesh_or_list, list):
            valid_meshes = [m for m in mesh_or_list if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
            if not valid_meshes:
                return None
            return trimesh.util.concatenate(valid_meshes)
        return mesh_or_list

    @staticmethod
    def generate_variable_step_spiral_2d(start_x, start_y, target_step_size, max_radius):

        # 生成 2D 动态步长螺旋点
        points = [[start_x, start_y]]
        b = target_step_size / (2 * np.pi)
        current_theta, current_r = 0.0, 0.0

        # 动态步长参数
        ramp_up_radius = target_step_size * 2.5
        min_step_ratio = 0.25

        max_iter = 50000
        count = 0

        while current_r < max_radius and count < max_iter:
            count += 1
            last_x, last_y = points[-1]
            found_next = False

            # 计算当前半径下的步长
            if current_r < ramp_up_radius:
                ratio = min_step_ratio + (1.0 - min_step_ratio) * (current_r / ramp_up_radius)
                actual_step = target_step_size * ratio
            else:
                actual_step = target_step_size

            # 迭代寻找下一个点
            for _ in range(20):
                if current_r < 1e-3:
                    step_theta = 0.1
                else:
                    step_theta = actual_step / current_r

                step_theta = min(step_theta, np.pi / 3)
                temp_theta = current_theta + step_theta
                temp_r = b * temp_theta

                temp_x = start_x + temp_r * np.cos(temp_theta)
                temp_y = start_y + temp_r * np.sin(temp_theta)
                dist = np.sqrt((temp_x - last_x) ** 2 + (temp_y - last_y) ** 2)

                if abs(dist - actual_step) < actual_step * 0.1:
                    current_theta = temp_theta
                    current_r = temp_r
                    points.append([temp_x, temp_y])
                    found_next = True
                    break

                # 修正逼近
                if dist > 0:
                    step_theta = step_theta * (actual_step / dist)
                    current_theta += step_theta
                    current_r = b * current_theta
                    points.append([start_x + current_r * np.cos(current_theta),
                                   start_y + current_r * np.sin(current_theta)])
                    found_next = True
                    break

            if not found_next:
                current_theta += 0.1

        return np.array(points)

    @staticmethod
    def reorder_segments_nearest_neighbor(points, normals, step_size, jump_tolerance):

        # 路径分段重连优化 (消除跨越空白区的跳变)
        if len(points) < 2: return points, normals
        segments_p, segments_n = [], []
        curr_seg_p, curr_seg_n = [points[0]], [normals[0]]
        threshold_sq = (step_size * jump_tolerance) ** 2

        # 1. 切分路段
        for i in range(1, len(points)):
            dist_sq = np.sum((points[i] - points[i - 1]) ** 2)
            if dist_sq > threshold_sq:
                segments_p.append(np.array(curr_seg_p))
                segments_n.append(np.array(curr_seg_n))
                curr_seg_p, curr_seg_n = [points[i]], [normals[i]]
            else:
                curr_seg_p.append(points[i])
                curr_seg_n.append(normals[i])
        segments_p.append(np.array(curr_seg_p))
        segments_n.append(np.array(curr_seg_n))

        # 2. 重连
        final_p, final_n = [segments_p[0]], [segments_n[0]]
        remaining = list(range(1, len(segments_p)))
        current_end = segments_p[0][-1]

        while remaining:
            best_dist = float('inf')
            best_idx = -1
            should_reverse = False

            for idx in remaining:
                seg = segments_p[idx]
                d_head = np.sum((seg[0] - current_end) ** 2)
                if d_head < best_dist:
                    best_dist, best_idx, should_reverse = d_head, idx, False
                d_tail = np.sum((seg[-1] - current_end) ** 2)
                if d_tail < best_dist:
                    best_dist, best_idx, should_reverse = d_tail, idx, True

            if best_idx != -1:
                p_add, n_add = segments_p[best_idx], segments_n[best_idx]
                if should_reverse: p_add, n_add = p_add[::-1], n_add[::-1]
                final_p.append(p_add)
                final_n.append(n_add)
                current_end = p_add[-1]
                remaining.remove(best_idx)
            else:
                break
        return np.vstack(final_p), np.vstack(final_n)

    @staticmethod
    def generate_zigzag_path(mesh, step_size, z_thresh):
        # 生成 Zigzag 路径
        bounds = mesh.bounds
        min_x, max_x = bounds[0, 0], bounds[1, 0]
        min_y, max_y = bounds[0, 1], bounds[1, 1]
        max_z = bounds[1, 2]

        x_range = np.arange(min_x, max_x + step_size, step_size)
        y_range = np.arange(min_y, max_y + step_size, step_size)
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        ray_origins = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, max_z + 10.0)])
        ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1))

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        if len(locations) == 0: return np.array([]), np.array([])

        normals = mesh.face_normals[index_tri]
        valid_mask = normals[:, 2] > z_thresh
        points, normals = locations[valid_mask], normals[valid_mask]

        if len(points) == 0: return points, normals

        # 排序逻辑
        quantized_x = np.round(points[:, 0] / (step_size * 0.1)) * (step_size * 0.1)
        sort_indices = np.lexsort((points[:, 1], quantized_x))
        sorted_points, sorted_normals = points[sort_indices], normals[sort_indices]
        quantized_x = quantized_x[sort_indices]
        unique_x = np.unique(quantized_x)

        final_p, final_n = [], []
        for i, x_val in enumerate(unique_x):
            mask = np.isclose(quantized_x, x_val)
            row_p, row_n = sorted_points[mask], sorted_normals[mask]
            if i % 2 == 1: row_p, row_n = row_p[::-1], row_n[::-1]
            final_p.append(row_p)
            final_n.append(row_n)

        if len(final_p) > 0: return np.vstack(final_p), np.vstack(final_n)
        return sorted_points, sorted_normals

    @staticmethod
    def filter_by_roi(points, normals, roi_dict):
        # 根据 ROI 过滤点
        mask = np.ones(len(points), dtype=bool)

        if roi_dict['x']['min'] is not None: mask &= (points[:, 0] >= roi_dict['x']['min'])
        if roi_dict['x']['max'] is not None: mask &= (points[:, 0] <= roi_dict['x']['max'])

        if roi_dict['y']['min'] is not None: mask &= (points[:, 1] >= roi_dict['y']['min'])
        if roi_dict['y']['max'] is not None: mask &= (points[:, 1] <= roi_dict['y']['max'])

        if roi_dict['z']['min'] is not None: mask &= (points[:, 2] >= roi_dict['z']['min'])
        if roi_dict['z']['max'] is not None: mask &= (points[:, 2] <= roi_dict['z']['max'])

        return points[mask], normals[mask]

    @staticmethod
    def compute_spiral_3d(mesh, center_x, center_y, radius, step_size, z_thresh):

        # 组合函数：生成2D螺旋 -> 投射3D -> 过滤法向 -> 优化重连
        # 1. 2D 螺旋
        pts_2d = CurvePathPlanner.generate_variable_step_spiral_2d(center_x, center_y, step_size, radius)

        # 2. 3D 投射
        max_z = mesh.bounds[1, 2] + 50
        origins = np.column_stack([pts_2d, np.full(len(pts_2d), max_z)])
        dirs = np.tile([0, 0, -1], (len(origins), 1))

        locs, idx_ray, idx_tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)

        if len(locs) == 0:
            return np.array([]), np.array([])

        # 3. 整理 & 法向过滤
        hit_map = {}
        all_n = mesh.face_normals[idx_tri]
        for i, r_idx in enumerate(idx_ray):
            if all_n[i][2] > z_thresh:
                hit_map[r_idx] = (locs[i], all_n[i])

        sp, sn = [], []
        # 按原始射线顺序提取，保证螺旋顺序
        for i in range(len(origins)):
            if i in hit_map:
                sp.append(hit_map[i][0])
                sn.append(hit_map[i][1])

        if not sp:
            return np.array([]), np.array([])

        # 4. 优化重连
        return CurvePathPlanner.reorder_segments_nearest_neighbor(
            np.array(sp), np.array(sn), step_size, jump_tolerance=2.0
        )