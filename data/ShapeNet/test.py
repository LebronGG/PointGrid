import os
import numpy as np
import glob
import tensorflow as tf
N = 16 # grid size is N x N x N
K = 4 # each cell has K points
NUM_CATEGORY = 4
NUM_SEG_PART = 5
NUM_PER_POINT_FEATURES = 3
NUM_FEATURES = K * NUM_PER_POINT_FEATURES + 1
def integer_label_to_one_hot_label(integer_label):
    if (len(integer_label.shape) == 0):
        one_hot_label = np.zeros((NUM_CATEGORY))
        one_hot_label[integer_label] = 1
    elif (len(integer_label.shape) == 1):
        one_hot_label = np.zeros((integer_label.shape[0], NUM_SEG_PART))
        for i in range(integer_label.shape[0]):
            one_hot_label[i, integer_label[i]] = 1
    elif (len(integer_label.shape) == 4):
        one_hot_label = np.zeros((N, N, N, K, NUM_SEG_PART))
        for i in range(N):
          for j in range(N):
            for k in range(N):
              for l in range(K):
                one_hot_label[i, j, k, l, integer_label[i, j, k, l]] = 1
    else:
        raise
    return one_hot_label

def pc2voxel(pc, pc_label):
    # Args:
    #     pc: size n x F where n is the number of points and F is feature size
    #     pc_label: size n x NUM_SEG_PART (one-hot encoding label)
    # Returns:
    #     voxel: N x N x N x K x (3+3)
    #     label: N x N x N x (K+1) x NUM_SEG_PART
    #     index: N x N x N x K

    num_points = pc.shape[0]
    data = np.zeros((N, N, N, NUM_FEATURES), dtype=np.float32)
    label = np.zeros((N, N, N, K+1, NUM_SEG_PART), dtype=np.float32)
    index = np.zeros((N, N, N, K), dtype=np.float32)
    xyz = pc[:, 0 : 3]
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz -= centroid
    xyz /= np.amax(np.sqrt(np.sum(xyz ** 2, axis=1)), axis=0) * 1.05
    idx = np.floor((xyz + 1.0) / 2.0 * N)
    L = [[] for _ in range(N * N * N)]
    for p in range(num_points):
        k = int(idx[p, 0] * N * N + idx[p, 1] * N + idx[p, 2])
        L[k].append(p)
    for i in range(N):
      for j in range(N):
        for k in range(N):
          u = int(i * N * N + j * N + k)
          if not L[u]:
              data[i, j, k, :] = np.zeros((NUM_FEATURES), dtype=np.float32)
              label[i, j, k, :, :] = 0
              label[i, j, k, :, 0] = 1
          elif (len(L[u]) >= K):
              choice = np.random.choice(L[u], size=K, replace=False)
              local_points = pc[choice, :] - np.array([-1.0 + (i + 0.5) * 2.0 / N, -1.0 + (j + 0.5) * 2.0 / N, -1.0 + (k + 0.5) * 2.0 / N], dtype=np.float32)
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(local_points, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
          else:
              choice = np.random.choice(L[u], size=K, replace=True)
              local_points = pc[choice, :] - np.array([-1.0 + (i + 0.5) * 2.0 / N, -1.0 + (j + 0.5) * 2.0 / N, -1.0 + (k + 0.5) * 2.0 / N], dtype=np.float32)
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(local_points, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
    return data, label, index

mat_content = np.load('./train/000004.npy')
pc = mat_content[:,0:3]
labels = np.squeeze(mat_content[: ,-2]).astype(int)
category = mat_content[:,-1].astype(int)
seg_label = integer_label_to_one_hot_label(labels)
# cat_label = integer_label_to_one_hot_label(category)
data, label, index = pc2voxel(pc, seg_label)
print(data.shape)
print(label.shape)
print(index.shape)

print(1)




