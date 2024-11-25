# %%
import numpy as np

def fbin_to_np(file_path, dtype=np.float32):
    with open(file_path, 'rb') as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=dtype, count=n * d)
    return data.reshape(n, d)

# load the vectors
vectors = fbin_to_np('data_dir/gist_base1B.fbin')
gt = fbin_to_np('data_dir/gist_ground-truth.bin', dtype=np.int32)

# load partitions
partitions = []
with open('/home/ben/gp-ann/data_dir/gist.partition.k=8.OGP.o=0.2') as f:
    for line in f:
        partitions.append(list(map(int, line.split())))
        
print("Partition sizes:\t", [len(p) for p in partitions])
print("     normalized:\t", [len(p) / (len(vectors) / len(partitions)) for p in partitions])


queries = fbin_to_np('data_dir/gist_query.fbin')

# construct the hierarchical k-means 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

def hierarchical_kmeans(vectors, included_indices, k, min_size=350):
    if len(included_indices) <= min_size:
        return []
    
    kmeans = KMeans(n_clusters=k).fit(vectors[included_indices])
    partition_indices = [np.where(kmeans.labels_ == i)[0] for i in range(k)]
    centers = [center for i, center in enumerate(kmeans.cluster_centers_) if len(partition_indices[i]) > min_size]
    
    for i, partition in enumerate(partition_indices):
        if len(partition) > min_size:
            centers.extend(hierarchical_kmeans(vectors, partition, k, min_size))
    
    return centers
# %%
print("computing centers")

partition_centers = []
for i in tqdm(range(len(partitions))):
    centers = hierarchical_kmeans(vectors, partitions[i], 8)
    partition_centers.append(centers)

combined_centers = np.concatenate(partition_centers)
breakpoints = np.cumsum([0] + [len(centers) for centers in partition_centers])

def map_center_to_partition(center_index):
    for i in range(len(breakpoints) - 1):
        if center_index < breakpoints[i + 1]:
            return i
    return
# %%
center_distances = cdist(queries, combined_centers, 'euclidean')

ten_nearest_centers = np.argsort(center_distances, axis=1)[:, :10]

print("partitions routed")
# how many of the vectors have their 10 nearest neighbors in the same partition?
ten_nn = gt[:, :10]
ten_nn_in_partition = np.zeros(len(vectors))
ten_nn_in_partition_2 = np.zeros(len(vectors))

partition_candidates = np.zeros(len(queries))



for i in range(len(queries)):
    cluster_neighbors = [map_center_to_partition(center) for center in ten_nearest_centers[i]]
    most_common_centers = np.argsort(np.bincount(cluster_neighbors))[::-1][:2]
    partition_candidates[i] = np.count_nonzero(np.bincount(cluster_neighbors))
    for j in range(10):
        if ten_nn[i, j] in partitions[most_common_centers[0]]:
            ten_nn_in_partition[i] += 1
        elif ten_nn[i, j] in partitions[most_common_centers[1]]:
            ten_nn_in_partition_2[i] += 1

print("percentage of 10 nearest neighbors in the same partition:\t", np.mean(ten_nn_in_partition / 10))
print("percentage of 10 nearest neighbors in the 2nd best partition:\t", np.mean(ten_nn_in_partition_2 / 10))

print("percentage of queries with 10 nearest neighbors in the 1st partition:\t", np.mean(ten_nn_in_partition == 10))
print("percentage of queries with 10 nearest neighbors in 1st and 2nd best partitions:\t", np.mean(ten_nn_in_partition + ten_nn_in_partition_2 == 10))


# %%
