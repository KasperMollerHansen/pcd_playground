import collections
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.sparse.csgraph import minimum_spanning_tree


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
        selected_dist = []
        min_dist = self.min_dist_factor * self.voxel_size
        for idx, dist in idx_dist:
            if dist < min_dist:
                continue

            dir_vec = points[idx] - centroid
            norm = np.linalg.norm(dir_vec)
            if norm == 0:
                continue
            dir_vec /= norm

            if selected_dirs:
                # Compute dot products with all selected directions
                dots = np.dot(selected_dirs, dir_vec)
                max_idx = np.argmax(dots)

                if dots[max_idx] > self.dot_threshold:
                    # ✅ Compare distance with the matching selected distance (not just last)
                    if np.isclose(dist, selected_dist[max_idx], atol= 2 * self.voxel_size):
                        # Require the candidate to be close to ALL selected points
                        if all(
                            np.linalg.norm(points[idx] - points[si]) > 4 * self.voxel_size
                            for si in selected_indices
                        ):
                            selected_indices.append(idx)
                            selected_dirs.append(dir_vec)
                            selected_dist.append(dist)
                            continue
                        else:
                            # Too close to an existing point → skip
                            continue
                    else:
                        # Direction too similar but distance not close → reject
                        continue

            # If we get here, either no similar direction or candidate passed checks
            selected_indices.append(idx)
            selected_dirs.append(dir_vec)
            selected_dist.append(dist)

            if len(selected_indices) >= self.max_edge_points:
                break

        return centroid, selected_indices


    def extract_edges_and_centroids(self, points, labels, best_k):
        edge_color = np.array([1, 0, 0])  # Red
        centroid_color = np.array([0, 0.5, 1])  # Blue-ish for centroid
        raw_edge_points = []
        raw_edge_clusters = []  # List of lists of cluster indices
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
                    raw_edge_clusters.append([k])
            raw_centroids.append(centroid)
        return np.array(raw_edge_points), raw_edge_clusters, np.array(raw_centroids), edge_color, centroid_color

    def merge_edge_points(self, raw_edge_points, raw_edge_clusters, points):
        if len(raw_edge_points) > 0:
            db = DBSCAN(eps=self.merge_radius_factor * self.voxel_size, min_samples=1).fit(raw_edge_points)
            merged_edge_points = []
            merged_clusters = []
            for label in np.unique(db.labels_):
                group = raw_edge_points[db.labels_ == label]
                group_clusters = [c for idx, c in enumerate(raw_edge_clusters) if db.labels_[idx] == label]
                # Flatten and deduplicate cluster indices
                merged_cluster = sorted(set(i for sublist in group_clusters for i in sublist))
                group_centroid = np.mean(group, axis=0)
                dists = np.linalg.norm(points - group_centroid, axis=1)
                best_idx = np.argmin(dists)
                merged_edge_points.append(points[best_idx])
                merged_clusters.append(merged_cluster)
            return np.array(merged_edge_points), merged_clusters
        else:
            return np.empty((0, 3)), []
        
    def merge_points_within_clusters(self, merged_edge_points, merged_clusters, points):
        """
        For each cluster, merge points that are within 2x merge_radius of each other.
        Returns new merged points and their cluster lists.
        """
        from sklearn.cluster import DBSCAN
        merged_edge_points = np.asarray(merged_edge_points)
        final_points = []
        final_clusters = []
        for k in set(i for clist in merged_clusters for i in clist):
            idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            if len(idxs) == 0:
                continue
            pts = merged_edge_points[idxs]
            db = DBSCAN(eps=2 * self.merge_radius_factor * self.voxel_size, min_samples=1).fit(pts)
            for label in np.unique(db.labels_):
                group = pts[db.labels_ == label]
                group_idxs = np.array(idxs)[db.labels_ == label]
                group_clusters = [merged_clusters[i] for i in group_idxs]
                merged_cluster = sorted(set(i for sublist in group_clusters for i in sublist))
                group_centroid = np.mean(group, axis=0)
                dists = np.linalg.norm(points - group_centroid, axis=1)
                best_idx = np.argmin(dists)
                final_points.append(points[best_idx])
                final_clusters.append(merged_cluster)
        return np.array(final_points), final_clusters
        
    def save_pointcloud_with_clusters(self, points, clusters, filename):
        """Save a point cloud with a custom cluster label list for each point (as .npz and .pcd)."""
        # Save as .npz for label persistence
        np.savez(filename + '.npz', points=points, clusters=np.array(clusters, dtype=object))
        # Save as .pcd for visualization (labels not preserved in .pcd)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # Color by number of clusters (for merged points, multi-cluster = blue, single = red)
        colors = np.array([[0,0,1] if len(c)>1 else [1,0,0] for c in clusters])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(filename + '.pcd', pcd)
        print(f"Saved {filename}.npz (with cluster lists) and {filename}.pcd (visualization)")

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
    
    def densify_skeleton(self, merged_edge_points, merged_clusters, points, labels, 
                        voxel_size, max_dist=5, min_points_for_skeleton=10):
        
        densified = {}
        
        for k in set(i for clist in merged_clusters for i in clist):
            # Get all points in this cluster
            cluster_mask = (labels == k)
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < min_points_for_skeleton:
                continue
                
            # Get edge points for this cluster
            idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            if len(idxs) < 2:
                continue
                
            edge_pts = np.array([merged_edge_points[i] for i in idxs])
            
            # Start with edge points as key skeleton points
            skel_points = set(tuple(pt) for pt in edge_pts)
            
            # Build minimum spanning tree of edge points
            n = len(edge_pts)
            dist_matrix = np.full((n, n), np.inf)
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(edge_pts[i] - edge_pts[j])
                    dist_matrix[i, j] = dist_matrix[j, i] = d
            
            mst = minimum_spanning_tree(dist_matrix)
            mst_edges = np.array(mst.nonzero()).T
            
            # Interpolate between all connected edge points
            for i, j in mst_edges:
                p1, p2 = edge_pts[int(i)], edge_pts[int(j)]
                d = np.linalg.norm(p2 - p1)
                n_steps = max(1, int(np.ceil(d / (max_dist * voxel_size))))
                for t in range(1, n_steps):
                    interp = p1 + (p2 - p1) * (t / n_steps)
                    skel_points.add(tuple(interp))
            
            if skel_points:
                densified[k] = np.array(list(skel_points))
        
        return densified
    
    def plot_densified_skeletons(self, points, densified, merged_edge_points, title="Complete Skeleton with Edge Points"):
        """
        Plot the complete skeleton including edge points as key skeleton points
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c='lightgray', s=2, alpha=0.3, label='All Points')
        
        colors = plt.cm.tab10.colors
        
        for k, skel_pts in densified.items():
            color = colors[k % len(colors)]
            
            # Plot all skeleton points (including edge points)
            ax.scatter(skel_pts[:, 0], skel_pts[:, 1], skel_pts[:, 2], 
                    c=color, s=40, alpha=0.8, label=f'Cluster {k} Skeleton')
            
            # Highlight the original edge points within the skeleton
            edge_mask = self._identify_edge_points_in_skeleton(skel_pts, merged_edge_points)
            if np.any(edge_mask):
                edge_pts = skel_pts[edge_mask]
                ax.scatter(edge_pts[:, 0], edge_pts[:, 1], edge_pts[:, 2], 
                        c=color, s=100, marker='o', edgecolors='black', 
                        linewidth=2.0, label=f'Cluster {k} Key Points')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self.set_axes_equal(ax)
        plt.title(title)
        plt.legend()
        plt.show()

    def _identify_edge_points_in_skeleton(self, skeleton_points, merged_edge_points, tol=1e-3):
        """
        Identify which skeleton points are original edge points
        """
        if len(skeleton_points) == 0 or len(merged_edge_points) == 0:
            return np.zeros(len(skeleton_points), dtype=bool)
        
        from sklearn.neighbors import NearestNeighbors
        
        # Find nearest edge point for each skeleton point
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(merged_edge_points)
        distances, _ = nn.kneighbors(skeleton_points)
        
        # Points that are very close to an edge point are considered edge points
        return distances.flatten() < tol

    def save_complete_skeleton(self, densified, merged_edge_points, merged_clusters, filename='complete_skeleton'):
        """
        Save the complete skeleton with edge points included and marked
        """
        all_skeleton_points = []
        cluster_labels = []
        is_edge_point = []  # Boolean indicating if point is an original edge point
        
        for k, skel_pts in densified.items():
            all_skeleton_points.extend(skel_pts)
            cluster_labels.extend([k] * len(skel_pts))
            
            # Identify which points in this cluster's skeleton are edge points
            cluster_edge_idxs = [i for i, clist in enumerate(merged_clusters) if k in clist]
            cluster_edge_pts = merged_edge_points[cluster_edge_idxs] if cluster_edge_idxs else np.array([])
            
            edge_mask = self._identify_edge_points_in_skeleton(skel_pts, cluster_edge_pts)
            is_edge_point.extend(edge_mask.tolist())
        
        all_skeleton_points = np.array(all_skeleton_points)
        is_edge_point = np.array(is_edge_point)
        
        # Save as NPZ with metadata
        np.savez(filename + '.npz', 
                skeleton_points=all_skeleton_points,
                cluster_labels=np.array(cluster_labels),
                is_edge_point=is_edge_point,
                edge_points=merged_edge_points)
        
        # Save visualization PCD with different colors for edge points
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(all_skeleton_points)
        
        # Color edge points red, other skeleton points blue
        colors = np.zeros((len(all_skeleton_points), 3))
        colors[is_edge_point] = [1, 0, 0]  # Red for edge points
        colors[~is_edge_point] = [0, 0, 1]  # Blue for other skeleton points
        
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filename + '.pcd', vis_pcd)
        print(f"Saved complete skeleton with {np.sum(is_edge_point)} edge points "
            f"and {np.sum(~is_edge_point)} medial points to {filename}.pcd")
        
    @staticmethod
    def check_duplicate_points(densified):
        all_points = []
        for cluster_pts in densified.values():
            all_points.extend(cluster_pts)
        
        all_points = np.array(all_points)
        unique_points, counts = np.unique(all_points, axis=0, return_counts=True)
        
        print(f"Total points in skeleton: {len(all_points)}")
        print(f"Unique points: {len(unique_points)}")
        print(f"Duplicate points: {np.sum(counts > 1)}")
        
        return unique_points, counts

    def main(self):
        pcd, points = self.load_point_cloud("static_cloud.pcd")
        labels, best_k, best_gmm = self.cluster_detection(points)
        raw_edge_points, raw_edge_clusters, raw_centroids, edge_color, centroid_color = self.extract_edges_and_centroids(points, labels, best_k)
        self.plot_point_cloud_with_edges(points, raw_edge_points, raw_centroids, edge_color, centroid_color,
            'Raw Cluster Edges (Red) and Centroids (Blue) in Point Cloud')
        merged_edge_points, merged_clusters = self.merge_edge_points(raw_edge_points, raw_edge_clusters, points)
        merged_edge_points, merged_clusters = self.merge_points_within_clusters(merged_edge_points, merged_clusters, points)
        self.save_pointcloud_with_clusters(raw_edge_points, raw_edge_clusters, 'edge_points_with_clusters')
        self.save_pointcloud_with_clusters(merged_edge_points, merged_clusters, 'merged_edge_points_with_clusters')
        self.save_merged_points(merged_edge_points, raw_centroids, edge_color, centroid_color)
        self.plot_point_cloud_with_edges(points, merged_edge_points, raw_centroids, edge_color, centroid_color,
            'Merged Cluster Edges (Red) and Centroids (Blue) in Point Cloud')
            # Densify skeletons so all segments <= 5*voxel_size
        # The merged edge points are now integrated into the skeleton
        densified = self.densify_skeleton(
            merged_edge_points, merged_clusters, points, labels, 
            self.voxel_size, max_dist=5
        )
        
        # Plot the complete skeleton with edge points included
        self.plot_densified_skeletons(
            points, densified, merged_edge_points,
            title="Complete Skeleton with Integrated Edge Points"
        )
        # Save the complete skeleton with edge points marked
        self.save_complete_skeleton(densified, merged_edge_points, merged_clusters)

        # Check for duplicate points
        unique_points, counts = self.check_duplicate_points(densified)

        # Print duplicate point information
        print(f"Total unique points after densification: {len(unique_points)}")
        print(f"Total duplicate points found: {np.sum(counts > 1)}")

#%%
skel = skeletonizer(voxel_size=1.0, super_voxel_factor=4.0,
                     max_edge_points=50, dot_threshold=0.8, min_dist_factor=10.0, max_clusters=20, merge_radius_factor=5.0)
skel.main()