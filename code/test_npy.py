import indoor3d_util
import tensorflow as tf
import numpy as np
import os
import sys
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

# DEFAULT SETTINGS
gpu_to_use = 0

# MAIN SCRIPT
batch_size = 1               # DO NOT CHANGE
purify = False                # Reassign label based on k-nearest neighbor. Set to False for large point cloud due to slow speed
knn = 5                      # for the purify

test_model=3
test_area=3

ckpt_dir = './log{}'.format(test_model)
output_dir = os.path.join(BASE_DIR, './log{}/test'.format(test_model))
flog = open(os.path.join(output_dir, 'log_test.txt'), 'w')
TESTING_FILE_LIST = './meta/area{}_data_label.txt'.format(test_area)
ROOM_PATH_LIST = [line.rstrip() for line in open(TESTING_FILE_LIST)]

test_data = []
test_sem = []
for room_path in ROOM_PATH_LIST:
    current_data, current_label = indoor3d_util.room2blocks_wrapper_normalized(room_path, model.NUM_POINT)
    current_data = current_data[:, :, :3]
    current_label = np.squeeze(current_label)
    test_data.append(current_data)
    test_sem.append(current_label)
test_data = np.concatenate(test_data, axis=0)
test_label = np.concatenate(test_sem, axis=0)
print 'test_model:', test_model
print 'test_area:', test_area
print 'test_data:', test_data.shape
print 'test_label:', test_label.shape


def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.NUM_FEATURES))
    seg_label_ph = tf.placeholder(tf.float32, shape=(batch_size, model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
    return pointgrid_ph, seg_label_ph


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

    with tf.device('/gpu:'+str(gpu_to_use)):
        pointgrid_ph,  seg_label_ph = placeholder_inputs()
        is_training_ph = tf.placeholder(tf.bool, shape=())

        pred_seg = model.get_model(pointgrid_ph, is_training=is_training_ph)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # Restore variables from disk.

        if not load_checkpoint(ckpt_dir, sess):
            exit()

        is_training = False
        gt_classes = [0 for _ in range(model.NUM_CATEGORY)]
        positive_classes = [0 for _ in range(model.NUM_CATEGORY)]
        true_positive_classes = [0 for _ in range(model.NUM_CATEGORY)]
        for i in range(test_data.shape[0]):
            pc = np.squeeze(test_data[i, :, :])
            labels = np.squeeze(test_label[i, :]).astype(int)
            seg_label = model.integer_label_to_one_hot_label(labels)
            pointgrid, pointgrid_label, index = model.pc2voxel(pc, seg_label)
            pointgrid = np.expand_dims(pointgrid, axis=0)
            pointgrid_label = np.expand_dims(pointgrid_label, axis=0)
            feed_dict = {
                pointgrid_ph: pointgrid,
                seg_label_ph: pointgrid_label,
                is_training_ph: is_training,
            }
            pred_seg_val = sess.run(pred_seg, feed_dict=feed_dict)
            pred_seg_val = pred_seg_val[0, :, :, :, :, :]
            pred_point_label = model.populateOneHotSegLabel(pc, pred_seg_val, index)

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
                gt_l = int(labels[j])
                pred_l = int(pred_point_label[j])
                gt_classes[gt_l] += 1
                positive_classes[pred_l] += 1
                true_positive_classes[gt_l] += int(gt_l == pred_l)


        printout(flog, 'gt_l count:{}'.format(sum(gt_classes)))
        printout(flog, 'positive_classes count:{}'.format(sum(positive_classes)))
        printout(flog, 'true_positive_classes count:{}'.format(sum(true_positive_classes)))
        printout(flog, 'gt_l count:{}'.format(gt_classes))
        printout(flog, 'positive_classes count:{}'.format(positive_classes))
        printout(flog, 'true_positive_classes count:{}'.format(true_positive_classes))

        iou_list = []
        for i in range(model.SEG_PART):
            try:
                iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                print '{}:{}'.format(i,iou)
            except ZeroDivisionError:
                iou = 0
                print '{}:{}'.format(i,iou)
            finally:
                iou_list.append(iou)
        printout(flog, 'IOU:{}'.format(iou_list))
        printout(flog, 'ACC:{}'.format(sum(true_positive_classes)*1.0 / (sum(positive_classes))))
        printout(flog, 'mIOU:{}'.format(sum(iou_list) / float(model.SEG_PART)))


with tf.Graph().as_default():
    predict()
