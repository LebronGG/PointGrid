import sys
import h5py
import numpy as np
import os
import glob
# import data_utils

root = './data'
output_folder='./val'
print(root)
folders = [(os.path.join(root,'val_data'), os.path.join(root,'val_label'))]
category_label_seg_max_dict = dict()
category_list = 0
for data_folder, label_folder in folders:
    print(data_folder)
    if not os.path.exists(data_folder):
        continue
    for filelist in sorted(os.listdir(data_folder)):
        data_filepath = os.path.join(data_folder, filelist)
        out_filename = filelist.split('.')[0]
        out_filename = out_filename + '.npy'
        out_filename=os.path.join(output_folder, out_filename)
        lable_filepath=data_filepath.replace('val_data','val_label')
        lable_filepath=lable_filepath.replace('pts','seg')
        points = np.loadtxt(data_filepath)
        part_label = np.loadtxt(lable_filepath).reshape(-1,1)
        if points.shape[0] !=part_label.shape[0]:
            print('file error')
            exit()
        print(points.shape)
        print(part_label.shape)
        data=np.concatenate((points,part_label),axis=1)
        np.save(out_filename, data)
        print(data.shape)


