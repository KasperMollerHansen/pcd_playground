import collections
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


class skeletonizer:
    def __init__(self, voxel_size=1.0, super_voxel_factor=4.0,
                 max_edge_points=10, dot_threshold=0.8, min_dist_factor=5.0, max_clusters=20,
                 merge_radius_factor=10.0):
        self.voxel_size = voxel_size
        self.super_voxel_size = super_voxel_factor * voxel_size
        self.max_edge_points = max_edge_points
        self.dot_threshold = dot_threshold
        self.min_dist_factor = min_dist_factor
        self.max_clusters = max_clusters
        self.merge_radius_factor = merge_radius_factor
        
    def quantize(self, x, y, z):
        return (int(np.floor(x / self.super_voxel_size)),
                int(np.floor(y / self.super_voxel_size)),
                int(np.floor(z / self.super_voxel_size)))
    
    def load_point_cloud(self, filename):
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        if len(pcd.colors) == 0:
            pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)))
        print(f"✅ Loaded point cloud with {len(points)} points.")
        return pcd, points
    
    def cluster_detection(self, points):
        print("🔄 Fitting GMM models (iterative elbow detection)...")
        bics, models = [], []
        threshold = 0.05  # 5% relative improvement cutoff
        elbow_idx = None
        for k in range(1, self.max_clusters + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
            gmm.fit(points)
            bics.append(gmm.bic(points))
            models.append(gmm)
            print(f"  k={k}: BIC={bics[-1]:.2f}")
            if k > 1:
                improvement = -(bics[-1] - bics[-2]) / abs(bics[-2])
                if improvement < threshold:
                    elbow_idx = k - 1
                    print(f"Elbow found at k={elbow_idx} (improvement={improvement:.4f} < {threshold})")
                    break
        if elbow_idx is None:
            elbow_idx = int(np.argmin(bics)) + 1
        bics = np.array(bics)
        best_k = elbow_idx
        best_gmm = models[best_k - 1]
        labels = best_gmm.predict(points)
        self.plot_elbow_curve(bics, best_k)
        print(f"Cluster sizes: {np.bincount(labels)}")
        return labels, best_k, best_gmm
    
    def plot_elbow_curve(self, bics, best_k):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(bics) + 1), bics, label='BIC', marker='o')
        plt.axvline(best_k, color='red', linestyle=':', label=f'Chosen k={best_k}')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('BIC score')
        plt.title('GMM BIC Curve with Automatic Elbow Detection')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

    def process_cluster(self, points):
        # 1. Supervoxel population map
        supervoxel_counts = collections.Counter(self.quantize(x, y, z) for x, y, z in points)
        # 2. Weighted centroid
        centroid = np.zeros(3)
        total_weight = 0.0
        for pt in points:
            v = self.quantize(*pt)
            weight = 1.0 / supervoxel_counts[v]
            centroid += weight * pt
            total_weight += weight
        if total_weight > 0:
            centroid /= total_weight
        # 3. Distances from centroid
        dists = np.linalg.norm(points - centroid, axis=1)
        idx_dist = sorted(enumerate(dists), key=lambda x: -x[1])
        # 4. Select up to max_edge_points with unique directions
        selected_indices = []
        selected_dirs = []
        min_dist = self.min_dist_factor * self.voxel_size
        for idx, dist in idx_dist:
            if dist < min_dist:
                continue
            dir_vec = points[idx] - centroid
            if np.linalg.norm(dir_vec) == 0:
                continue
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            if any(np.dot(dir_vec, sel_dir) > self.dot_threshold for sel_dir in selected_dirs):
                continue
            selected_indices.append(idx)
            selected_dirs.append(dir_vec)
            if len(selected_indices) >= self.max_edge_points:
                break
        return centroid, selected_indices

    def extract_edges_and_centroids(self, points, labels, best_k):
        edge_color = np.array([1, 0, 0])  # Red
        centroid_color = np.array([0, 0.5, 1])  # Blue-ish for centroid
        raw_edge_points = []
        raw_centroids = []
        for k in range(best_k):
            cluster_mask = (labels == k)
            cluster_points = points[cluster_mask]
            if len(cluster_points) == 0:
                continue
            centroid, edge_indices = self.process_cluster(cluster_points)
            for ei in edge_indices:
                if 0 <= ei < len(cluster_points):
                    raw_edge_points.append(cluster_points[ei])
            raw_centroids.append(centroid)
        return np.array(raw_edge_points), np.array(raw_centroids), edge_color, centroid_color

    def merge_edge_points(self, raw_edge_points, points):
        if len(raw_edge_points) > 0:
            db = DBSCAN(eps=self.merge_radius_factor * self.voxel_size, min_samples=1).fit(raw_edge_points)
            merged_edge_points = []
            for label in np.unique(db.labels_):
                group = raw_edge_points[db.labels_ == label]
                group_centroid = np.mean(group, axis=0)
                dists = np.linalg.norm(points - group_centroid, axis=1)
                best_idx = np.argmin(dists)
                merged_edge_points.append(points[best_idx])
            return np.array(merged_edge_points)
        else:
            return np.empty((0, 3))

    def save_merged_points(self, merged_edge_points, raw_centroids, edge_color, centroid_color, filename='merged_edges_and_centroids.pcd'):
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(np.vstack([merged_edge_points, raw_centroids]))
        vis_pcd.colors = o3d.utility.Vector3dVector(
            np.vstack([
                np.tile(edge_color, (len(merged_edge_points), 1)),
                np.tile(centroid_color, (len(raw_centroids), 1))
            ])
        )
        o3d.io.write_point_cloud(filename, vis_pcd)
        print(f"Saved {filename} with merged edge and centroid points.")

    def plot_point_cloud_with_edges(self, points, edge_points, centroids, edge_color, centroid_color, title):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightgray', s=2, label='All Points')
        if len(edge_points) > 0:
            ax.scatter(edge_points[:, 0], edge_points[:, 1], edge_points[:, 2],
                       c=[edge_color], s=40, label='Edge Points')
        if len(centroids) > 0:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c=[centroid_color], s=100, marker='*', label='Centroids')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self.set_axes_equal(ax)
        plt.title(title)
        ax.legend()
        plt.show()

    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    def main(self):
        pcd, points = self.load_point_cloud("static_cloud.pcd")
        labels, best_k, best_gmm = self.cluster_detection(points)
        raw_edge_points, raw_centroids, edge_color, centroid_color = self.extract_edges_and_centroids(points, labels, best_k)
        self.plot_point_cloud_with_edges(points, raw_edge_points, raw_centroids, edge_color, centroid_color,
            'Raw Cluster Edges (Red) and Centroids (Blue) in Point Cloud')
        merged_edge_points = self.merge_edge_points(raw_edge_points, points)
        self.save_merged_points(merged_edge_points, raw_centroids, edge_color, centroid_color)
        self.plot_point_cloud_with_edges(points, merged_edge_points, raw_centroids, edge_color, centroid_color,
            'Merged Cluster Edges (Red) and Centroids (Blue) in Point Cloud')



#%%
skel = skeletonizer(voxel_size=1.0, super_voxel_factor=4.0,
                     max_edge_points=10, dot_threshold=0.95, min_dist_factor=10.0, max_clusters=20, merge_radius_factor=5.0)
skel.main()