import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
import glob
from skimage import measure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()


# DEFAULT SETTINGS
gpu_to_use = 0
output_dir = os.path.join(BASE_DIR, './test_results')

# MAIN SCRIPT
batch_size = 1               # DO NOT CHANGE
purify = True                # Reassign label based on k-nearest neighbor. Set to False for large point cloud due to slow speed
knn = 5                      # for the purify

def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]

TESTING_FILE_LIST = [get_file_name(file_name) for file_name in glob.glob('../data/ShapeNet/val/' + '*.npy')]


color_map = json.load(open('part_color_mapping.json', 'r'))

lines = [line.rstrip('\n') for line in open('sphere.txt')]
nSphereVertices = int(lines[0])
sphereVertices = np.zeros((nSphereVertices, 3))
for i in range(nSphereVertices):
    coordinates = lines[i + 1].split()
    for j in range(len(coordinates)):
        sphereVertices[i, j] = float(coordinates[j])
nSphereFaces = int(lines[nSphereVertices + 1])
sphereFaces = np.zeros((nSphereFaces, 3))
for i in range(nSphereFaces):
    indices = lines[i + nSphereVertices + 2].split()
    for j in range(len(coordinates)):
        sphereFaces[i, j] = int(indices[j])

def output_color_point_cloud(data, seg, out_file, r=0.01):
    count = 0
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            for j in range(nSphereVertices):
                f.write('v %f %f %f %f %f %f\n' % \
                        (data[i][0] + sphereVertices[j][0] * r, data[i][1] + sphereVertices[j][1] * r, data[i][2] + sphereVertices[j][2] * r, color[0], color[1], color[2]))
            for j in range(nSphereFaces):
                f.write('f %d %d %d\n' % (count + sphereFaces[j][0], count + sphereFaces[j][1], count + sphereFaces[j][2]))
            count += nSphereVertices

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.NUM_FEATURES))
    seg_label_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
    return pointgrid_ph, seg_label_ph

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

def predict():
    is_training = False
    
    with tf.device('/gpu:'+str(gpu_to_use)):
        pointgrid_ph,  seg_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        pred_seg = model.get_model(pointgrid_ph, is_training=is_training_ph)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        flog = open(os.path.join(output_dir, 'log.txt'), 'w')

        # Restore variables from disk.
        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            exit()

        if not os.path.exists('../data/ShapeNet/test-PointGrid'):
            os.mkdir('../data/ShapeNet/test-PointGrid')

        test_file_idx = np.arange(0, len(TESTING_FILE_LIST))
        np.random.shuffle(test_file_idx)

        gt_classes = [0 for _ in range(model.NUM_CATEGORY)]
        negative_classes = [0 for _ in range(model.NUM_CATEGORY)]
        positive_classes = [0 for _ in range(model.NUM_CATEGORY)]
        for loop in range(len(TESTING_FILE_LIST)):
            mat_content = np.load('../data/ShapeNet/' + TESTING_FILE_LIST[test_file_idx[loop]] + '.npy')
            choice=np.random.choice(mat_content.shape[0],model.SAMPLE_NUM, replace=False)
            mat_content=mat_content[choice,:]

            xyz = mat_content[:, 0:3]
            xyz = model.rotate_pc(xyz)
            rgb = mat_content[:, 3:6] / 255.0

            pc = np.concatenate((xyz, rgb), axis=1)
            labels = np.squeeze(mat_content[:, -1]).astype(int)

            seg_label = model.integer_label_to_one_hot_label(labels)
            pointgrid, pointgrid_label, _ = model.pc2voxel(pc, seg_label)
            print(TESTING_FILE_LIST[test_file_idx[loop]])

            pointgrid = np.expand_dims(pointgrid, axis=0)
            pointgrid_label = np.expand_dims(pointgrid_label, axis=0)
            feed_dict = {
                         pointgrid_ph: pointgrid,
                         seg_label_ph: pointgrid_label,
                         is_training_ph: is_training,
                        }
            pred_seg_val = sess.run(pred_seg, feed_dict = feed_dict)
            #    pred_seg: of size B x N x N x N x (K+1) x NUM_PART_SEG

            pred_seg_val = pred_seg_val[0, :, :, :, :, :]
            pred_point_label = model.populateOneHotSegLabel(pc, pred_seg_val, index)
            #     pred_point_label: size n x 1
            if purify == True:
                pre_label = pred_point_label
                for i in range(pc.shape[0]): #one point cloud num 2500--2800
                    idx = np.argsort(np.sum((pc[i, :] - pc) ** 2, axis=1))
                    j, L = 0, []
                    for _ in range(knn):
                        if (idx[j] == i):
                            j += 1
                        L.append(pre_label[idx[j]])
                        j += 1
                    majority = max(set(L), key=L.count)
                    if (pre_label[i] == 0 or len(set(L)) == 1):
                        pred_point_label[i] = majority

            for j in range(pred_point_label.shape[0]):
                gt_classes[labels[j]]+=1
                if int(labels[j])==int(pred_point_label[j]):
                    positive_classes[labels[j]]+=1
                else:
                    negative_classes[labels[j]]+=1
                print('negative:{},positive:{},gt_classes:{}'.format(negative_classes,positive_classes,gt_classes))
        print('negative_classes count:',negative_classes)
        print('positive_classes count:',positive_classes)
        print('gt_classes count:',gt_classes)

        iou_list=[]
        for i in range(model.NUM_CATEGORY):
            iou = positive_classes[i] / gt_classes[i]
            iou_list.append(iou)
        print('IOU:',iou_list)
        print('ACC:',sum(positive_classes)/sum(gt_classes))
        print('mIOU:',sum(iou_list) / 13.0)

with tf.Graph().as_default():
    predict()
