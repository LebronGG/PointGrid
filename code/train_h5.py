import argparse
import tensorflow as tf
import threading
import numpy as np
from datetime import datetime
import os
import sys
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

i=6
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=24, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=100, help='Epoch to run [default: 200]')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--log_dir', default='log{}'.format(i), help='Log dir [default: log]')
parser.add_argument('--input_train', type=str, default='data/train_hdf5_file_list_Area{}.txt'.format(i), help='Input train data')
parser.add_argument('--input_test', type=str, default='data/test_hdf5_file_list_Area{}.txt'.format(i), help='Input test data')
parser.add_argument('--restore_model', type=str, help='Pretrained model')
FLAGS = parser.parse_args()

# MAIN SCRIPT
LEARNING_RATE = 1e-4
TRAINING_EPOCHES = FLAGS.epoch
BATCH_SIZE = FLAGS.batch
LOG_DIR = FLAGS.log_dir
TRAINING_FILE_LIST = FLAGS.input_train
TESTING_FILE_LIST = FLAGS.input_test

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


print('#### Batch Size: {0}'.format(BATCH_SIZE))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))


def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]


def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

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

# Load train data
train_file_list = getDataFiles(TRAINING_FILE_LIST)
train_data = []
train_sem = []
for h5_filename in train_file_list:
    cur_data, cur_sem = loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
    train_data.append(cur_data)
    train_sem.append(cur_sem)
train_data = np.concatenate(train_data, axis=0)
train_label = np.concatenate(train_sem, axis=0)
print('train_data:', train_data.shape)
print('train_label:', train_label.shape)
# Load val data
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


def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE,model.N, model.N, model.N, model.NUM_FEATURES))
    seg_label_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE,model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
    return pointgrid_ph,seg_label_ph

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



def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointgrid_ph, seg_label_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            # model
            pred_seg = model.get_model(pointgrid_ph, is_training=is_training_ph)
            total_loss, seg_loss = model.get_loss(pred_seg, seg_label_ph)

            # optimization
            total_var = tf.trainable_variables()
            step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss, var_list=total_var)

        # write logs to the disk
        flog = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())

        # Add summary writers
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        fcmd = open(os.path.join(LOG_DIR, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(epoch_num,sess, train_writer):
            is_training = True
            current_data, current_label, _ = shuffle_data(train_data, train_label)
            num_data = train_data.shape[0]
            num_batch = num_data // BATCH_SIZE
            total_loss_acc = 0.0
            seg_loss_acc = 0.0
            display_mark = max([num_batch // 4, 1])

            for batch_idx in range(num_batch):
                if batch_idx % 100 == 0:
                    print('Current batch/total batch num: %d/%d' % (batch_idx, num_batch))
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE

                pointgrid,pointgrid_label= transfor_data(current_data[start_idx:end_idx, :, :],current_label[start_idx:end_idx,:])

                feed_dict = {is_training_ph: is_training,
                             pointgrid_ph: pointgrid,
                             seg_label_ph: pointgrid_label}

                _ , total_loss_val, seg_loss_val = sess.run([step, total_loss, seg_loss], feed_dict = feed_dict)
                # train_writer.add_summary(total_loss_val,seg_loss_val)
                total_loss_acc += total_loss_val
                seg_loss_acc += seg_loss_val

                if ((i+1) % display_mark == 0):
                    printout(flog, 'Epoch %d/%d - Iter %d/%d' % (epoch_num+1, TRAINING_EPOCHES, i+1, num_batch))
                    printout(flog, 'Total Loss: %f' % (total_loss_acc / (i+1)))
                    printout(flog, 'Segmentation Loss: %f' % (seg_loss_acc / (i+1)))

            printout(flog, '\tMean Total Loss: %f' % (total_loss_acc / num_batch))
            printout(flog, '\tMean Segmentation Loss: %f' % (seg_loss_acc / num_batch))

        def test_one_epoch(sess, test_writer):
            is_training = False
            current_data, current_label, _ = shuffle_data(test_data, test_label)
            num_data = test_data.shape[0]
            num_batch = num_data // BATCH_SIZE
            total_loss_acc = 0.0
            seg_loss_acc = 0.0


            for batch_idx in range(num_batch):
                if batch_idx % 100 == 0:
                    print('Current batch/total batch num: %d/%d' % (batch_idx, num_batch))
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE

                pointgrid,pointgrid_label= transfor_data(current_data[start_idx:end_idx, :, :],current_label[start_idx:end_idx,:])

                feed_dict = {is_training_ph: is_training,
                             pointgrid_ph: pointgrid,
                             seg_label_ph: pointgrid_label}

                _ , total_loss_val, seg_loss_val = sess.run([step, total_loss, seg_loss], feed_dict=feed_dict)
                # test_writer.add_summary(step, total_loss_val, seg_loss_val)
                total_loss_acc += total_loss_val
                seg_loss_acc += seg_loss_val

            printout(flog, '\tMean Total Loss: %f' % (total_loss_acc / num_batch))
            printout(flog, '\tMean Segmentation Loss: %f' % (seg_loss_acc / num_batch))


        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch+1, TRAINING_EPOCHES))

            train_one_epoch(epoch,sess, train_writer)
            test_one_epoch(sess, test_writer)

            if epoch % 5 == 0:
                cp_filename = saver.save(sess, os.path.join(LOG_DIR, 'epoch_' + str(epoch)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()
        flog.close()

if __name__=='__main__':
    train()
