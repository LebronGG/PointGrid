import argparse
import tensorflow as tf
import numpy as np
import os
import sys
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model


i=6

TESTING_FILE_LIST = 'data/test_hdf5_file_list_Area{}.txt'.format(i)
def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_groupseglabel_stanfordindoor(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    if 'label' in f:
        label = f['label'][:].astype(np.int32)
    else :
        label = []
        print ('label ins None')
    return (data[:,:,:3], label)

def transfor_data(cur_data,cur_sem):
    data=[]
    label=[]
    for i in range(cur_data.shape[0]):
        pc = np.squeeze(cur_data[i, :, :])
        labels = np.squeeze(cur_sem[i, :]).astype(int)
        seg_label = model.integer_label_to_one_hot_label(labels)
        pointgrid, pointgrid_label, index = model.pc2voxel(pc, seg_label)
        data.append(pointgrid)
        label.append(pointgrid_label)
    data = np.asarray(data)
    label = np.asarray(label)
    return data,label

test_file_list = getDataFiles(TESTING_FILE_LIST)
test_data = []
test_sem = []
for h5_filename in test_file_list:
    cur_data, cur_sem = loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
    test_data.append(cur_data)
    test_sem.append(cur_sem)
test_data = np.concatenate(test_data, axis=0)
test_label = np.concatenate(test_sem, axis=0)
print('test_data:', test_data.shape)
print('test_label:', test_label.shape)


pointgrid,pointgrid_label=transfor_data(test_data[:30,:,:],test_label[:30,:])
print pointgrid.shape
print pointgrid_label.shape









