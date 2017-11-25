import os
import numpy as np
import tensorflow as tf
import time
from conv_def import conv_bn_relu, deconv_bn_relu, conv3d, deconv3d
from data_io import load_image_and_label, get_image_and_label_batch
from glob import glob
from json_io import dict_to_json, json_to_dict
from loss_def import dice_loss_function, softmax_loss_function

''' 3D U-Net Model '''


class Unet3D(object):
    def __init__(self, sess, parameter_dict):
        # member variables
        self.dice_loss_coefficient = 0.1
        self.l2_loss_coefficient = 0.001
        
        self.input_image = None
        self.input_ground_truth = None
        self.predicted_prob = None
        self.predicted_label = None
        self.auxiliary1_prob_1x = None
        self.auxiliary2_prob_1x = None
        self.auxiliary3_prob_1x = None
        self.main_dice_loss = None
        self.auxiliary1_dice_loss = None
        self.auxiliary2_dice_loss = None
        self.auxiliary3_dice_loss = None
        self.total_dice_loss = None
        self.main_weight_loss = None
        self.auxiliary1_weight_loss = None
        self.auxiliary2_weight_loss = None
        self.auxiliary3_weight_loss = None
        self.total_weight_loss = None
        self.total_loss = None
        self.trainable_variables = None
        self.log_writer = None
        self.fine_tuning_variables = None
        self.saver = None
        self.saver_fine_tuning = None
        self.l2_loss = None

        # predefined
        # single-gpu
        self.gpu_number = len(parameter_dict['gpu'].split(','))
        if self.gpu_number > 1:
            self.device = ['/gpu:0', '/gpu:1', '/cpu:0']
        else:
            self.device = ['/gpu:0', '/gpu:0', '/cpu:0']
        self.sess = sess
        self.parameter_dict = parameter_dict
        self.phase = parameter_dict['phase']
        self.batch_size = parameter_dict['batch_size']
        self.input_size = parameter_dict['input_size']
        self.input_channels = parameter_dict['input_channels']
        self.output_size = parameter_dict['output_size']
        self.output_channels = parameter_dict['output_channels']
        self.learning_rate = parameter_dict['learning_rate']
        self.beta1 = parameter_dict['beta1']
        self.epoch = parameter_dict['epoch']
        self.train_data_dir = parameter_dict['train_data_dir']
        self.test_data_dir = parameter_dict['test_data_dir']
        self.label_data_dir = parameter_dict['label_data_dir']
        self.prediction_dir = parameter_dict['prediction_dir']
        self.model_name = parameter_dict['model_name']
        self.name_with_runtime = parameter_dict['name_with_runtime']
        self.checkpoint_dir = parameter_dict['checkpoint_dir']
        self.resize_coefficient = parameter_dict['resize_coefficient']
        self.test_stride = parameter_dict['test_stride']

        # scalable number of feature maps: default 32
        self.feat_num = parameter_dict['feature_number']
        self.index_start = parameter_dict['index_start']
        self.index_included = parameter_dict['index_included']

        # from previous version
        self.save_interval = parameter_dict['save_interval']
        self.test_interval = parameter_dict['test_interval']
        self.cube_overlapping_factor = parameter_dict['cube_overlapping_factor']

        # build model
        self.build_model()

    def unet_model(self, inputs):
        is_training = (self.phase == 'train')
        concat_dimension = 4  # channels_last

        # down-sampling path
        # device: gpu0
        with tf.device(device_name_or_function=self.device[0]):
            # first level
            encoder1_1 = conv_bn_relu(inputs=inputs, output_channels=self.feat_num, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder1_1')
            encoder1_2 = conv_bn_relu(inputs=encoder1_1, output_channels=self.feat_num*2, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder1_2')
            pool1 = tf.layers.max_pooling3d(
                inputs=encoder1_2,
                pool_size=2,                    # pool_depth, pool_height, pool_width
                strides=2,
                padding='valid',                # No padding, default
                data_format='channels_last',    # default
                name='pool1'
            )
            # second level
            encoder2_1 = conv_bn_relu(inputs=pool1, output_channels=self.feat_num*2, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder2_1')
            encoder2_2 = conv_bn_relu(inputs=encoder2_1, output_channels=self.feat_num*4, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder2_2')
            pool2 = tf.layers.max_pooling3d(inputs=encoder2_2, pool_size=2, strides=2, name='pool2')
            # third level
            encoder3_1 = conv_bn_relu(inputs=pool2, output_channels=self.feat_num*4, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder3_1')
            encoder3_2 = conv_bn_relu(inputs=encoder3_1, output_channels=self.feat_num*8, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder3_2')
            pool3 = tf.layers.max_pooling3d(inputs=encoder3_2, pool_size=2, strides=2, name='pool3')
            # forth level
            encoder4_1 = conv_bn_relu(inputs=pool3, output_channels=self.feat_num*8, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder4_1')
            encoder4_2 = conv_bn_relu(inputs=encoder4_1, output_channels=self.feat_num*16, kernel_size=3, stride=1,
                                      is_training=is_training, name='encoder4_2')
            bottom = encoder4_2

        # up-sampling path
        # device: gpu1
        with tf.device(device_name_or_function=self.device[1]):
            # third level
            deconv3 = deconv_bn_relu(inputs=bottom, output_channels=self.feat_num*16, is_training=is_training,
                                     name='deconv3')
            concat_3 = tf.concat([deconv3, encoder3_2], axis=concat_dimension, name='concat_3')
            decoder3_1 = conv_bn_relu(inputs=concat_3, output_channels=self.feat_num*8, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder3_1')
            decoder3_2 = conv_bn_relu(inputs=decoder3_1, output_channels=self.feat_num*8, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder3_2')
            # second level
            deconv2 = deconv_bn_relu(inputs=decoder3_2, output_channels=self.feat_num*8, is_training=is_training,
                                     name='deconv2')
            concat_2 = tf.concat([deconv2, encoder2_2], axis=concat_dimension, name='concat_2')
            decoder2_1 = conv_bn_relu(inputs=concat_2, output_channels=self.feat_num*4, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder2_1')
            decoder2_2 = conv_bn_relu(inputs=decoder2_1, output_channels=self.feat_num*4, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder2_2')
            # first level
            deconv1 = deconv_bn_relu(inputs=decoder2_2, output_channels=self.feat_num*4, is_training=is_training,
                                     name='deconv1')
            concat_1 = tf.concat([deconv1, encoder1_2], axis=concat_dimension, name='concat_1')
            decoder1_1 = conv_bn_relu(inputs=concat_1, output_channels=self.feat_num*2, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder1_1')
            decoder1_2 = conv_bn_relu(inputs=decoder1_1, output_channels=self.feat_num*2, kernel_size=3, stride=1,
                                      is_training=is_training, name='decoder1_2')
            feature = decoder1_2
            # predicted probability
            predicted_prob = conv3d(inputs=feature, output_channels=self.output_channels, kernel_size=1,
                                    stride=1, use_bias=True, name='predicted_prob')

            '''auxiliary prediction'''
            # forth level
            auxiliary3_prob_8x = conv3d(inputs=encoder4_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary3_prob_8x')
            auxiliary3_prob_4x = deconv3d(inputs=auxiliary3_prob_8x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_4x')
            auxiliary3_prob_2x = deconv3d(inputs=auxiliary3_prob_4x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_2x')
            auxiliary3_prob_1x = deconv3d(inputs=auxiliary3_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary3_prob_1x')
            # third level
            auxiliary2_prob_4x = conv3d(inputs=decoder3_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary2_prob_4x')
            auxiliary2_prob_2x = deconv3d(inputs=auxiliary2_prob_4x, output_channels=self.output_channels,
                                          name='auxiliary2_prob_2x')
            auxiliary2_prob_1x = deconv3d(inputs=auxiliary2_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary2_prob_1x')
            # second level
            auxiliary1_prob_2x = conv3d(inputs=decoder2_2, output_channels=self.output_channels, kernel_size=1,
                                        stride=1, use_bias=True, name='auxiliary1_prob_2x')
            auxiliary1_prob_1x = deconv3d(inputs=auxiliary1_prob_2x, output_channels=self.output_channels,
                                          name='auxiliary1_prob_1x')
        # TODO: draw a graph

        # device: cpu0
        with tf.device(device_name_or_function=self.device[2]):
            softmax_prob = tf.nn.softmax(logits=predicted_prob, name='softmax_prob')
            predicted_label = tf.argmax(input=softmax_prob, axis=4, name='predicted_label')

        return predicted_prob, predicted_label, auxiliary1_prob_1x, auxiliary2_prob_1x, auxiliary3_prob_1x

    def build_model(self):
        # input data and labels
        self.input_image = tf.placeholder(dtype=tf.float32,
                                          shape=[self.batch_size, self.input_size, self.input_size,
                                                 self.input_size, self.input_channels], name='input_image')
        self.input_ground_truth = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.input_size,
                                                                        self.input_size, self.input_size],
                                                 name='input_ground_truth')
        # probability
        self.predicted_prob, self.predicted_label, self.auxiliary1_prob_1x, \
            self.auxiliary2_prob_1x, self.auxiliary3_prob_1x = self.unet_model(self.input_image)

        # dice loss
        self.main_dice_loss = dice_loss_function(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_dice_loss = dice_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_dice_loss = dice_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_dice_loss = dice_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_dice_loss = \
            self.main_dice_loss + \
            self.auxiliary1_dice_loss * 0.8 + \
            self.auxiliary2_dice_loss * 0.4 + \
            self.auxiliary3_dice_loss * 0.2
        # class-weighted cross-entropy loss
        self.main_weight_loss = softmax_loss_function(self.predicted_prob, self.input_ground_truth)
        self.auxiliary1_weight_loss = softmax_loss_function(self.auxiliary1_prob_1x, self.input_ground_truth)
        self.auxiliary2_weight_loss = softmax_loss_function(self.auxiliary2_prob_1x, self.input_ground_truth)
        self.auxiliary3_weight_loss = softmax_loss_function(self.auxiliary3_prob_1x, self.input_ground_truth)
        self.total_weight_loss = \
            self.main_weight_loss +\
            self.auxiliary1_weight_loss * 0.9 + \
            self.auxiliary2_weight_loss * 0.6 + \
            self.auxiliary3_weight_loss * 0.3

        # regularization
        _norm = 0
        tensor_name_dict = json_to_dict('regularization.json', read_file=True)
        for tensor_name in tensor_name_dict.values():
            _tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
            _norm += tf.nn.l2_loss(_tensor)
        self.l2_loss = _norm

        # total loss
        self.total_loss = \
            self.total_dice_loss * self.dice_loss_coefficient + self.total_weight_loss + \
            self.l2_loss * self.l2_loss_coefficient

        # trainable variables
        self.trainable_variables = tf.trainable_variables()

        # TODO: how to extract layers for fine-tuning? why?
        '''How to list all of them'''
        fine_tuning_layer = [
                'encoder1_1/encoder1_1_conv/kernel:0',
                'encoder1_2/encoder1_2_conv/kernel:0',
                'encoder2_1/encoder2_1_conv/kernel:0',
                'encoder2_2/encoder2_2_conv/kernel:0',
                'encoder3_1/encoder3_1_conv/kernel:0',
                'encoder3_2/encoder3_2_conv/kernel:0',
                'encoder4_1/encoder4_1_conv/kernel:0',
                'encoder4_2/encoder4_2_conv/kernel:0',
        ]

        # TODO: what does this part mean
        self.fine_tuning_variables = []
        for variable in self.trainable_variables:
            # print('\'%s\',' % variable.name)
            for index, kernel_name in enumerate(fine_tuning_layer):
                if kernel_name in variable.name:
                    self.fine_tuning_variables.append(variable)
                    break  # not necessary to continue

        self.saver = tf.train.Saver(max_to_keep=20)
        self.saver_fine_tuning = tf.train.Saver(self.fine_tuning_variables)
        # The Saver class adds ops to save and restore variables to and from checkpoints.
        # It also provides convenience methods to run these ops.
        print('Model built successfully.')

    def save_checkpoint(self, checkpoint_dir, model_name, global_step):
        model_dir = '%s_%s_%s_s%s-%s' % (self.feat_num, self.batch_size, self.output_size,
                                         self.index_start, self.index_included)
        '''Why?'''
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        # defaults to the list of all saveable objects

    # TODO: load check point -> To be checked!
    def load_checkpoint(self, checkpoint_dir):
        model_dir = '%s_%s_%s_s%s-%s' % (self.feat_num, self.batch_size, self.output_size,
                                         self.index_start, self.index_included)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        # A CheckpointState if the state was available, None otherwise.
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False

    '''A function for fine-tuning'''

    def train(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(
            self.total_loss, var_list=self.trainable_variables
        )

        # initialization
        variables_initialization = tf.global_variables_initializer()
        self.sess.run(variables_initialization)

        # TODO: load pre-trained model
        # TODO: load checkpoint

        # save log
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        self.log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.sess.graph)

        # load all volume files
        image_list = glob(pathname='{}/*.nii.gz'.format(self.train_data_dir))
        label_list = glob(pathname='{}/*.nii.gz'.format(self.label_data_dir))
        image_list.sort()
        label_list.sort()
        image_data_list_full, label_data_list_full = load_image_and_label(
            image_list, label_list, self.resize_coefficient)
        # dictionary passing for test
        image_label_dict = dict()
        image_label_dict['image_data_list_full'] = image_data_list_full
        image_label_dict['label_data_list_full'] = label_data_list_full
        # sample selection
        index_start = self.index_start
        index_excluded = self.index_included + 1
        image_data_list = image_data_list_full[index_start:index_excluded]
        label_data_list = label_data_list_full[index_start:index_excluded]
        print('Selected samples: ', image_list[index_start:index_excluded],
              label_list[index_start:index_excluded])
        print('Data loaded successfully.')

        if not os.path.exists('loss/'):
            os.makedirs('loss/')
        line_buffer = 1
        with open(file='loss/loss_'+self.name_with_runtime+'.txt', mode='w', buffering=line_buffer) as loss_log:
            loss_log.write('[Train Mode]\n')
            loss_log.write(dict_to_json(self.parameter_dict))
            loss_log.write('\n')

            for epoch in np.arange(self.epoch):
                start_time = time.time()

                # load batch
                train_data_batch, train_label_batch = get_image_and_label_batch(
                    image_data_list, label_data_list, self.input_size, self.batch_size,
                    rotation_flag=True, flip_flag=True)
                # val_data_batch, val_label_batch = get_image_and_label_batch(
                #     image_data_list, label_data_list, self.input_size, self.batch_size)
                '''The same data at this stage'''

                # update network
                _, train_loss, dice_loss, weight_loss, l2_loss = self.sess.run(
                    [optimizer, self.total_loss, self.total_dice_loss, self.total_weight_loss, self.l2_loss],
                    feed_dict={self.input_image: train_data_batch,
                               self.input_ground_truth: train_label_batch})
                '''Summary'''
                # TODO: test do not run each time
                # val_loss = self.total_loss.eval({self.input_image: val_data_batch,
                #                                  self.input_ground_truth: val_label_batch})
                # val_prediction = self.sess.run(self.predicted_label,
                #                                feed_dict={self.input_image: val_data_batch})

                # loss_log.write(str(np.unique(val_label_batch)))
                # loss_log.write(str(np.unique(val_prediction)))
                string_format = f'[label] {str(np.unique(train_label_batch))} \n'
                loss_log.write(string_format)
                print(string_format, end='')

                # TODO: Dice? Problem?
                # dice = []
                # for i in range(self.output_channels):
                #     intersection = np.sum(
                #         ((val_label_batch[:, :, :, :] == i) * 1) * ((val_prediction[:, :, :, :] == i) * 1)
                #     )
                #     union = np.sum(
                #         ((val_label_batch[:, :, :, :] == i) * 1) + ((val_prediction[:, :, :, :] == i) * 1)
                #     ) + 1e-5
                #     '''Why not necessary to square'''
                #     dice.append(2.0 * intersection / union)
                # loss_log.write('[Dice] %s \n' % dice)

                # loss_log.write('%s %s\n' % (train_loss, val_loss))
                # deprecated: remove validation loss, dice loss original
                output_format = '[Epoch] %d, time: %4.4f, train_loss: %.8f \n' \
                                '[Loss] dice_loss: %.8f, weight_loss: %.8f, l2_loss: %.8f \n\n'\
                                % (epoch+1, time.time() - start_time, train_loss,
                                   dice_loss, weight_loss, l2_loss)
                loss_log.write(output_format)
                print(output_format, end='')
                if np.mod(epoch+1, self.save_interval) == 0:
                    self.save_checkpoint(self.checkpoint_dir, self.model_name, global_step=epoch+1)
                    print('[Save] Model saved with epoch %d' % (epoch+1))
                # TODO: test
                if np.mod(epoch + 1, self.test_interval) == 0:
                    self.test(initialization=False, image_label_dict=image_label_dict, train_epoch=epoch+1)

    # May produce memory issue - not test
    def test(self, initialization=True, image_label_dict=None, train_epoch=None):
        image_data_list_full = None
        label_data_list_full = None
        if not initialization and image_label_dict is None:
            print(' [!] Fatal Error: no feeding data')
            return
        if initialization:
            # initialization
            variables_initialization = tf.global_variables_initializer()
            self.sess.run(variables_initialization)

            # TODO: load pre-trained model
            # TODO: load checkpoint
            if self.load_checkpoint(self.checkpoint_dir):
                print(" [*] Load Success")
            else:
                print(" [!] Load Failed")
                exit(1)  # exit with load error

            # save log
            if not os.path.exists('logs/'):
                os.makedirs('logs/')
            self.log_writer = tf.summary.FileWriter(logdir='logs/', graph=self.sess.graph)

            # load all volume files
            image_list = glob(pathname='{}/*.nii.gz'.format(self.train_data_dir))
            label_list = glob(pathname='{}/*.nii.gz'.format(self.label_data_dir))
            image_list.sort()
            label_list.sort()
            # no selection on test
            image_data_list_full, label_data_list_full = load_image_and_label(
                image_list, label_list, self.resize_coefficient)
            print('Data loaded successfully.')
        else:
            image_data_list_full = image_label_dict['image_data_list_full']
            label_data_list_full = image_label_dict['label_data_list_full']

        if not os.path.exists('test/'):
            os.makedirs('test/')
        line_buffer = 1
        # appending
        with open(file='test/test_'+self.name_with_runtime+'.txt', mode='a', buffering=line_buffer) as test_log:
            if train_epoch is None:
                test_log.write('[Test Mode]\n')
                test_log.write(dict_to_json(self.parameter_dict))
                test_log.write('\n')
            else:
                print(f'========== [Test] Epoch {train_epoch} ==========\n')
                test_log.write(f'========== [Test] Epoch {train_epoch} ==========\n')

            for ith_sample in range(len(image_data_list_full)):
                sample_start_time = time.time()
                test_batch_size = 1
                test_image_data = image_data_list_full[ith_sample]
                # output_channels -> number of class
                # input_channels -> 1
                ith_depth, ith_height, ith_width = test_image_data.shape
                # final_test_prediction_full = np.zeros([ith_depth, ith_height, ith_width], dtype='int32')
                test_prediction_full_with_count = np.zeros(
                    [test_batch_size, ith_depth, ith_height, ith_width, self.output_channels], dtype='int32')
                test_data_full = np.reshape(image_data_list_full[ith_sample].astype('float32'), [
                    test_batch_size, ith_depth, ith_height, ith_width, self.input_channels])
                test_label_full = np.reshape(label_data_list_full[ith_sample].astype('int32'), [
                    test_batch_size, ith_depth, ith_height, ith_width])

                # TODO: boundary to be considered!
                depth_range = np.arange(ith_depth - self.input_size + 1, step=self.test_stride)
                height_range = np.arange(ith_height - self.input_size + 1, step=self.test_stride)
                width_range = np.arange(ith_width - self.input_size + 1, step=self.test_stride)
                print(depth_range, height_range, width_range)
                # reverse order
                depth_range_reverse = [ith_depth - reverse_depth - self.input_size for reverse_depth in depth_range]
                height_range_reverse = [ith_height - reverse_height - self.input_size for reverse_height in height_range]
                width_range_reverse = [ith_width - reverse_width - self.input_size for reverse_width in width_range]

                depth_range = list(set(np.append(depth_range, depth_range_reverse)))
                height_range = list(set(np.append(height_range, height_range_reverse)))
                width_range = list(set(np.append(width_range, width_range_reverse)))
                depth_range.sort()
                height_range.sort()
                width_range.sort()
                print(depth_range, height_range, width_range)

                for d in depth_range:
                    for h in height_range:
                        for w in width_range:
                            batch_start_time = time.time()
                            test_data_batch = test_data_full[
                                               test_batch_size-1,
                                               d:d + self.input_size,
                                               h:h + self.input_size,
                                               w:w + self.input_size,
                                               self.input_channels-1
                                               ]
                            # numpy property of shape of broadcasting
                            test_data_batch = np.reshape(
                                test_data_batch, [self.batch_size, self.input_size, self.input_size,
                                                  self.input_size, self.input_channels])

                            test_label_batch = test_label_full[
                                               test_batch_size-1,
                                               d:d + self.input_size,
                                               h:h + self.input_size,
                                               w:w + self.input_size
                                               ]
                            test_label_batch = np.reshape(
                                test_label_batch, [
                                    self.batch_size, self.input_size, self.input_size, self.input_size])

                            predict_label_batch = test_prediction_full_with_count[
                                                  test_batch_size-1,
                                                  d:d + self.input_size,
                                                  h:h + self.input_size,
                                                  w:w + self.input_size,
                                                  :
                                                  ]
                            predict_label_batch = np.reshape(
                                predict_label_batch, [self.batch_size, self.input_size, self.input_size,
                                                      self.input_size, self.output_channels])

                            # update network
                            test_prediction_batch = self.sess.run(self.predicted_label,
                                                                  feed_dict={self.input_image: test_data_batch})
                            # TODO: add loss described in the paper
                            test_loss, dice_loss, weight_loss, l2_loss = self.sess.run(
                                [self.total_loss, self.total_dice_loss, self.total_weight_loss, self.l2_loss],
                                feed_dict={self.input_image: test_data_batch,
                                           self.input_ground_truth: test_label_batch})

                            '''Update record of prediction_full_batch'''
                            for _label in range(3):
                                _batch, _depth, _height, _width = np.where(test_prediction_batch == _label)
                                predict_label_batch[_batch, _depth, _height, _width, _label] = \
                                    predict_label_batch[_batch, _depth, _height, _width, _label] + 1

                            '''Extract information of loss'''
                            # Necessary to print info during test??

                            unique = f'[label] {str(np.unique(test_label_batch))} ' \
                                     f'{str(np.unique(test_prediction_batch))}\n'
                            test_log.write(unique)
                            # print(unique)

                            # loss_log.write('%s %s\n' % (train_loss, val_loss))
                            output_format = '[DHW] %f, %f, %f\n' \
                                            '[Sample] %d, time: %4.4f, test_loss: %.8f \n' \
                                            '[Loss] dice_loss: %.8f, weight_loss: %.8f, l2_loss: %.8f \n\n' \
                                            % (d/depth_range[-1], h/height_range[-1], w/width_range[-1],
                                               ith_sample, time.time() - batch_start_time, test_loss,
                                               dice_loss, weight_loss, l2_loss)
                            test_log.write(output_format)
                            # print(output_format, end='')

                '''Voting'''
                final_test_prediction_full = np.argmax(test_prediction_full_with_count, axis=4)  # axis=-1
                if final_test_prediction_full.shape != test_label_full.shape:
                    print('Dimension mismatch of labels!')
                    exit(1)
                # voting finished -> labels
                if not os.path.exists(self.prediction_dir):
                    os.makedirs(self.prediction_dir)
                # TODO: store the prediction label

                '''Revise the name of dice calculation'''
                # Dice and Jaccard
                dice = []
                jaccard = []
                volumetric_overlap_error = []
                volume_difference = []
                for classify_label in range(self.output_channels):
                    _ground = (test_label_full[:, :, :, :] == classify_label) * 1
                    _result = (final_test_prediction_full[:, :, :, :] == classify_label) * 1
                    # intersection
                    intersection = np.sum(_result * _ground)
                    # summation
                    _addition = _ground + _result
                    summation_result = np.sum(_result)
                    summation_ground = np.sum(_ground)
                    summation = summation_result + summation_ground + 1e-5
                    # union
                    # TODO: use np.logical_and(_ground, _addition) -> faster
                    union = np.sum((_addition[:, :, :, :] > 0) * 1) + 1e-5
                    # appending
                    dice.append(2.0 * intersection / summation)
                    jaccard.append(intersection / union)
                    volumetric_overlap_error.append(1 - intersection / union)
                    volume_difference.append((summation_result - summation_ground) / summation_ground)

                # Accuracy overall
                correct_count = np.sum((test_label_full == final_test_prediction_full) * 1)
                total_count = test_batch_size * ith_depth * ith_height * ith_width
                accuracy = correct_count / total_count
                accuracy_with_label = []
                for _label in range(3):
                    _correct = np.sum(
                        (
                            ((test_label_full == final_test_prediction_full) * 1) *
                            ((final_test_prediction_full == _label) * 1)
                        ))
                    _total = np.sum((test_label_full == _label) * 1)
                    accuracy_with_label.append([_correct/_total, _correct, _total])

                output_format = f'[Dice] Sample: {ith_sample}, Dice: {dice}\n' \
                                f'[Jaccard] Sample: {ith_sample}, Jaccard: {jaccard}\n' \
                                f'[VOE] Sample: {ith_sample}, VOE: {volumetric_overlap_error}\n' \
                                f'[VD] Sample: {ith_sample}, VD: {volume_difference}\n' \
                                f'[Time] {time.time() - sample_start_time}\n' \
                                f'[Accuracy] {accuracy} {accuracy_with_label}\n\n'
                print(output_format, end='')
                test_log.write(output_format)


if __name__ == '__main__':
    sess = tf.Session()
