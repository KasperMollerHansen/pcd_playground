import collections
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# --- Load your actual point cloud ---

pcd = o3d.io.read_point_cloud("static_cloud.pcd")
points = np.asarray(pcd.points)
# Ensure the point cloud has a color array (white) if missing
if len(pcd.colors) == 0:
    pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3)))

print(f"✅ Loaded point cloud with {len(points)} points.")

# --- Normalize points for better GMM stability ---

# --- Fit GMMs for a range of k ---
max_k = 20
bics, aics, models = [], [], []

print("🔄 Fitting GMM models (iterative elbow detection)...")
bics, aics, models = [], [], []
threshold = 0.05  # 5% relative improvement cutoff
elbow_idx = None
for k in range(1, max_k + 1):
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(points)
    bics.append(gmm.bic(points))
    aics.append(gmm.aic(points))
    models.append(gmm)
    print(f"  k={k}: BIC={bics[-1]:.2f}")
    if k > 1:
        improvement = -(bics[-1] - bics[-2]) / abs(bics[-2])
        if improvement < threshold:
            elbow_idx = k - 1
            print(f"Elbow found at k={elbow_idx} (improvement={improvement:.4f} < {threshold})")
            break

if elbow_idx is None:
    # fallback to minimum BIC if no clear elbow found
    elbow_idx = int(np.argmin(bics))

bics = np.array(bics)
aics = np.array(aics)
best_k = elbow_idx
best_gmm = models[best_k - 1]
print(f"\n✅ Best number of clusters (automatic elbow): k={best_k}")

# --- Plot BIC/AIC curves ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(bics) + 1), bics, label='BIC', marker='o')
plt.plot(range(1, len(aics) + 1), aics, label='AIC', marker='x', linestyle='--')
plt.axvline(best_k, color='red', linestyle=':', label=f'Chosen k={best_k}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.title('GMM BIC/AIC Curve with Automatic Elbow Detection')
plt.legend()
plt.grid()
plt.show(block=False)

# --- Predict cluster labels for each point ---
# --- Predict cluster labels for each point ---
labels = best_gmm.predict(points)
print(f"Cluster sizes: {np.bincount(labels)}")

# --- Edge and centroid detection for each cluster ---
def quantize(x, y, z, voxel_size):
    return (int(np.floor(x / voxel_size)),
            int(np.floor(y / voxel_size)),
            int(np.floor(z / voxel_size)))

def process_cluster(points, voxel_size, super_voxel_factor=4.0, max_edge_points=10, dot_threshold=0.8, min_dist_factor=5.0):
    super_voxel_size = super_voxel_factor * voxel_size
    # 1. Supervoxel population map
    supervoxel_counts = collections.Counter(quantize(x, y, z, super_voxel_size) for x, y, z in points)
    # 2. Weighted centroid
    centroid = np.zeros(3)
    total_weight = 0.0
    for pt in points:
        v = quantize(*pt, super_voxel_size)
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
    min_dist = min_dist_factor * voxel_size
    for idx, dist in idx_dist:
        if dist < min_dist:
            continue
        dir_vec = points[idx] - centroid
        if np.linalg.norm(dir_vec) == 0:
            continue
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        if any(np.dot(dir_vec, sel_dir) > dot_threshold for sel_dir in selected_dirs):
            continue
        selected_indices.append(idx)
        selected_dirs.append(dir_vec)
        if len(selected_indices) >= max_edge_points:
            break
    return centroid, selected_indices

o3d.io.write_point_cloud('clustered_with_edges.pcd', pcd)

# --- Only plot/save edge points and centroids ---
# --- Only plot/save merged edge points and centroids ---
voxel_size = 1.0
edge_color = np.array([1, 0, 0])  # Red
centroid_color = np.array([0, 0.5, 1])  # Blue-ish for centroid

raw_edge_points = []
raw_centroids = []
for k in range(best_k):
    cluster_mask = (labels == k)
    cluster_points = points[cluster_mask]
    if len(cluster_points) == 0:
        continue
    centroid, edge_indices = process_cluster(
        cluster_points, voxel_size,
        max_edge_points=10, dot_threshold=0.95, min_dist_factor=10.0
    )
    # Add edge points
    for ei in edge_indices:
        if 0 <= ei < len(cluster_points):
            raw_edge_points.append(cluster_points[ei])
    # Add centroid
    raw_centroids.append(centroid)

raw_edge_points = np.array(raw_edge_points)
raw_centroids = np.array(raw_centroids)


# --- Plot BEFORE merging ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightgray', s=2, label='All Points')
if len(raw_edge_points) > 0:
    ax.scatter(raw_edge_points[:, 0], raw_edge_points[:, 1], raw_edge_points[:, 2],
               c='red', s=40, label='Raw Edge Points')
if len(raw_centroids) > 0:
    ax.scatter(raw_centroids[:, 0], raw_centroids[:, 1], raw_centroids[:, 2],
               c=[centroid_color], s=100, marker='*', label='Centroids')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
def set_axes_equal(ax):
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
set_axes_equal(ax)
plt.title('Raw Cluster Edges (Red) and Centroids (Blue) in Point Cloud')
ax.legend()
plt.show()

# --- Merge edge points within 10xvoxel_size using DBSCAN ---
if len(raw_edge_points) > 0:
    db = DBSCAN(eps=10*voxel_size, min_samples=1).fit(raw_edge_points)
    merged_edge_points = []
    for label in np.unique(db.labels_):
        group = raw_edge_points[db.labels_ == label]
        # Use group centroid
        group_centroid = np.mean(group, axis=0)
        # Find closest point in original point cloud
        dists = np.linalg.norm(points - group_centroid, axis=1)
        best_idx = np.argmin(dists)
        merged_edge_points.append(points[best_idx])
    merged_edge_points = np.array(merged_edge_points)
else:
    merged_edge_points = np.empty((0, 3))

# Save merged edge points and centroids only
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(np.vstack([merged_edge_points, raw_centroids]))
vis_pcd.colors = o3d.utility.Vector3dVector(
    np.vstack([
        np.tile(edge_color, (len(merged_edge_points), 1)),
        np.tile(centroid_color, (len(raw_centroids), 1))
    ])
)
o3d.io.write_point_cloud('merged_edges_and_centroids.pcd', vis_pcd)
print("Saved merged_edges_and_centroids.pcd with merged edge and centroid points.")

# Visualize merged: real PCD in gray, merged edges in red, centroids in blue
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightgray', s=2, label='All Points')
if len(merged_edge_points) > 0:
    ax.scatter(merged_edge_points[:, 0], merged_edge_points[:, 1], merged_edge_points[:, 2],
               c='red', s=60, label='Merged Edge Points')
if len(raw_centroids) > 0:
    ax.scatter(raw_centroids[:, 0], raw_centroids[:, 1], raw_centroids[:, 2],
               c=[centroid_color], s=100, marker='*', label='Centroids')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
def set_axes_equal(ax):
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
set_axes_equal(ax)
plt.title('Merged Cluster Edges (Red) and Centroids (Blue) in Point Cloud')
ax.legend()
plt.show()