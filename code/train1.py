import argparse
import subprocess
import tensorflow as tf
import threading
import numpy as np
import h5py
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
import network as model

# DEFAULT SETTINGS
i=6
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--output_dir', default='train_results', help='Log dir [default: log]')
parser.add_argument('--input_train', type=str, default='data/train_hdf5_file_list_Area{}.txt'.format(i), help='Input train data')
parser.add_argument('--input_test', type=str, default='data/test_hdf5_file_list_Area{}.txt'.format(i), help='Input test data')
parser.add_argument('--restore_model', type=str, help='Pretrained model')
FLAGS = parser.parse_args()

# MAIN SCRIPT
batch_size = FLAGS.batch
output_dir = FLAGS.output_dir
TRAINING_FILE_LIST = FLAGS.input_train
TESTING_FILE_LIST = FLAGS.input_test




if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print('#### Batch Size: {0}'.format(batch_size))
print('#### Training using GPU: {0}'.format(FLAGS.gpu))

LEARNING_RATE = 1e-4
TRAINING_EPOCHES = FLAGS.epoch
print('### Training epoch: {0}'.format(TRAINING_EPOCHES))

def get_file_name(file_path):
    parts = file_path.split('/')
    part = parts[-1]
    parts = part.split('.')
    return parts[0]


MODEL_STORAGE_PATH = os.path.join(output_dir, 'trained_models')
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

LOG_STORAGE_PATH = os.path.join(output_dir, 'logs')
if not os.path.exists(LOG_STORAGE_PATH):
    os.mkdir(LOG_STORAGE_PATH)

SUMMARIES_FOLDER =  os.path.join(output_dir, 'summaries')
if not os.path.exists(SUMMARIES_FOLDER):
    os.mkdir(SUMMARIES_FOLDER)

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

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


def load_and_enqueue(sess, enqueue_op, pointgrid_ph, seg_label_ph):
    for i in range(train_data.shape[0]):
        pc = np.squeeze(train_data[i,:,:])
        labels = np.squeeze(train_label[i,:]).astype(int)
        seg_label = model.integer_label_to_one_hot_label(labels)
        pointgrid, pointgrid_label, _ = model.pc2voxel(pc, seg_label)
        sess.run(enqueue_op, feed_dict={pointgrid_ph: pointgrid, seg_label_ph: pointgrid_label})

def placeholder_inputs():
    pointgrid_ph = tf.placeholder(tf.float32, shape=(model.N, model.N, model.N, model.NUM_FEATURES))
    seg_label_ph = tf.placeholder(tf.float32, shape=(model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART))
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

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, target=None, args=None):
        super(StoppableThread, self).__init__(target=target, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointgrid_ph, seg_label_ph = placeholder_inputs()
            is_training_ph = tf.placeholder(tf.bool, shape=())

            queue = tf.FIFOQueue(capacity=20*batch_size, dtypes=[tf.float32, tf.float32],\
                                                         shapes=[[model.N, model.N, model.N, model.NUM_FEATURES],\
                                                                 [model.N, model.N, model.N, model.K+1, model.NUM_SEG_PART]])
            enqueue_op = queue.enqueue([pointgrid_ph, seg_label_ph])
            dequeue_pointgrid, dequeue_seg_label = queue.dequeue_many(batch_size)

            # model
            pred_seg = model.get_model(dequeue_pointgrid, is_training=is_training_ph)

            # loss
            total_loss, seg_loss = model.get_loss(pred_seg, dequeue_seg_label)


            # optimization
            total_var = tf.trainable_variables()
            step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss, var_list=total_var)

        # write logs to the disk
        flog = open(os.path.join(LOG_STORAGE_PATH, 'log_train.txt'), 'w')

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        ckpt_dir = './train_results/trained_models'
        if not load_checkpoint(ckpt_dir, sess):
            sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

        fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
        fcmd.write(str(FLAGS))
        fcmd.close()

        def train_one_epoch(epoch_num):
            is_training = True

            num_data = train_data.shape[0]
            num_batch = num_data // batch_size
            total_loss_acc = 0.0
            seg_loss_acc = 0.0
            display_mark = max([num_batch // 4, 1])
            for i in range(num_batch):
                _, total_loss_val,  seg_loss_val = sess.run([step, total_loss, seg_loss], feed_dict={is_training_ph: is_training})
                total_loss_acc += total_loss_val
                seg_loss_acc += seg_loss_val

                if ((i+1) % display_mark == 0):
                    printout(flog, 'Epoch %d/%d - Iter %d/%d' % (epoch_num+1, TRAINING_EPOCHES, i+1, num_batch))
                    printout(flog, 'Total Loss: %f' % (total_loss_acc / (i+1)))
                    printout(flog, 'Segmentation Loss: %f' % (seg_loss_acc / (i+1)))

            printout(flog, '\tMean Total Loss: %f' % (total_loss_acc / num_batch))
            printout(flog, '\tMean Segmentation Loss: %f' % (seg_loss_acc / num_batch))

        def test_one_epoch(epoch_num):
            is_training = False
            total_loss_acc = 0.0
            seg_loss_acc = 0.0
            gt_classes = [0 for _ in range(model.NUM_CATEGORY)]
            positive_classes = [0 for _ in range(model.NUM_CATEGORY)]
            true_positive_classes = [0 for _ in range(model.NUM_CATEGORY)]
            for i in range(test_data.shape[0]):
                pc = np.squeeze(test_data[i, :, :])
                labels = np.squeeze(test_label[i, :]).astype(int)
                seg_label = model.integer_label_to_one_hot_label(labels)
                pointgrid, pointgrid_label, index = model.pc2voxel(pc, seg_label)
                feed_dict = {is_training_ph: is_training,
                             pointgrid_ph: pointgrid,
                             seg_label_ph: pointgrid_label}
                total_loss_val,  seg_loss_val,pred_seg_val= sess.run([total_loss, seg_loss,pred_seg], feed_dict=feed_dict)
                total_loss_acc += total_loss_val
                seg_loss_acc += seg_loss_val

                pred_seg_val = pred_seg_val[0, :, :, :, :, :]
                pred_point_label = model.populateOneHotSegLabel(pc, pred_seg_val, index)
                for j in range(pred_point_label.shape[0]):
                    gt_l = int(labels[j])
                    pred_l = int(pred_point_label[j])
                    gt_classes[gt_l - 1] += 1
                    positive_classes[pred_l - 1] += 1
                    true_positive_classes[gt_l - 1] += int(gt_l == pred_l)

            printout(flog, 'gt_l count:{}'.format(gt_classes))
            printout(flog, 'positive_classes count:{}'.format(positive_classes))
            printout(flog, 'true_positive_classes count:{}'.format(true_positive_classes))

            iou_list = []
            for i in range(model.SEG_PART):
                iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
                iou_list.append(iou)
            printout(flog, 'IOU:{}'.format(iou_list))
            printout(flog, 'ACC:{}'.format(sum(true_positive_classes) / sum(positive_classes)))
            printout(flog, 'mIOU:{}'.format(sum(iou_list) / float(model.SEG_PART)))
            printout(flog, '\tMean Total Loss: %f' % (total_loss_acc / test_data.shape[0]))
            printout(flog, '\tMean Segmentation Loss: %f' % (seg_loss_acc / test_data.shape[0]))

        if not os.path.exists(MODEL_STORAGE_PATH):
            os.mkdir(MODEL_STORAGE_PATH)

        coord = tf.train.Coordinator()
        for num_thread in range(16):
            t = StoppableThread(target=load_and_enqueue, args=(sess, enqueue_op, pointgrid_ph, seg_label_ph))
            t.setDaemon(True)
            t.start()
            coord.register_thread(t)

        for epoch in range(TRAINING_EPOCHES):
            printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch+1, TRAINING_EPOCHES))

            train_one_epoch(epoch)
            # test_one_epoch(epoch)

            if (epoch+1) % 1 == 0:
                cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch+1)+'.ckpt'))
                printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

            flog.flush()
        flog.close()

if __name__=='__main__':
    train()
