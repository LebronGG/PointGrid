import os
import numpy as np
import glob
import tensorflow as tf
N = 16 # grid size is N x N x N
K = 4 # each cell has K points
NUM_CATEGORY = 13
NUM_SEG_PART = NUM_CATEGORY+1
NUM_PER_POINT_FEATURES = 6
NUM_FEATURES = K * NUM_PER_POINT_FEATURES + 1
SAMPLE_NUM=4096

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
              local_points = pc[choice, :] - np.ones(NUM_PER_POINT_FEATURES)*(np.float32(-1.0 + (i + 0.5) * 2.0 / N))
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(local_points, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
          else:
              choice = np.random.choice(L[u], size=K, replace=True)
              local_points = pc[choice, :] - np.ones(NUM_PER_POINT_FEATURES) * (np.float32(-1.0 + (i + 0.5) * 2.0 / N))
              data[i, j, k, 0 : K * NUM_PER_POINT_FEATURES] = np.reshape(local_points, (K * NUM_PER_POINT_FEATURES))
              data[i, j, k, K * NUM_PER_POINT_FEATURES] = 1.0
              label[i, j, k, 0 : K, :] = pc_label[choice, :]
              majority = np.argmax(np.sum(pc_label[L[u], :], axis=0))
              label[i, j, k, K, :] = 0
              label[i, j, k, K, majority] = 1
              index[i, j, k, :] = choice
    return data, label, index

def rotate_pc(pc):
    # Args:
    #     pc: size n x 3
    # Returns:
    #     rotated_pc: size n x 3
    xyz=pc[:,:3]
    rgb=pc[:,3:6]
    angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    xyz = np.dot(xyz, rotation_matrix)
    rotated_pc=np.concatenate((xyz,rgb),axis=1)
    return rotated_pc

mat_content = np.load('./val/Area_1_conferenceRoom_1.npy')
choice = np.random.choice(mat_content.shape[0], size=SAMPLE_NUM, replace=True)
mat_content = mat_content[choice, :]
print(mat_content.shape)
xyz = mat_content[:, :3]
rgb = mat_content[:, 3:6] / 256
pc = np.concatenate((xyz, rgb), axis=1)
labels = np.squeeze(mat_content[:, -1]).astype(int)

pc = rotate_pc(pc)
seg_label = integer_label_to_one_hot_label(labels)
pointgrid, pointgrid_label, _ = pc2voxel(pc, seg_label)


print(pointgrid.shape)
print(pointgrid_label.shape)






