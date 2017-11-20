import argparse
import datetime
import os
import tensorflow as tf
from json_io import dict_to_json
from model import Unet3D

''' Main Function '''


def init_parameter(name):
    # dictionary
    parameter_dict = dict()
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    parameter_dict['phase'] = "train"
    parameter_dict['batch_size'] = 1
    parameter_dict['input_size'] = 64
    parameter_dict['input_channels'] = 1
    parameter_dict['output_size'] = 64
    parameter_dict['output_channels'] = 3
    parameter_dict['learning_rate'] = 0.001
    parameter_dict['beta1'] = 0.5
    parameter_dict['epoch'] = 10000  # 100000 -> 10000
    parameter_dict['train_data_dir'] = "../hvsmr/data/"
    parameter_dict['test_data_dir'] = "../hvsmr/data/"
    parameter_dict['label_data_dir'] = "../hvsmr/label"
    parameter_dict['model_name'] = "hvsmr_" + name + ".model"
    parameter_dict['name_with_runtime'] = name
    parameter_dict['checkpoint_dir'] = "checkpoint/"
    parameter_dict['resize_coefficient'] = 1.0
    parameter_dict['test_stride'] = 32  # for overlap
    # from previous version
    parameter_dict['save_interval'] = 1000  # 10000 -> 1000
    parameter_dict['cube_overlapping_factor'] = 4
    parameter_dict['gpu'] = '0,1'

    # scalable number of feature maps: default 32
    parameter_dict['feature_number'] = 16  # 32 -> 16
    parameter_dict['index_start'] = 7
    parameter_dict['index_included'] = 7

    return parameter_dict


# What is the input parameter
def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', help='cuda visible devices')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--sample', help='sample selection')
    args = parser.parse_args()
    if args.gpu:
        gpu = args.gpu
    else:
        gpu = '0,1'

    # set cuda visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # name the model
    if args.test:
        name = 'test'
    else:
        name = 'train'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = name + '_' + current_time

    # load predefined training data
    parameter_dict = init_parameter(name)
    if args.gpu:
        parameter_dict['gpu'] = args.gpu
    if args.test:
        parameter_dict['phase'] = 'test'
    if args.sample:
        sample_select = args.sample.strip().split(',')
        if len(sample_select) == 1:
            parameter_dict['index_start'] = int(sample_select[0])
            parameter_dict['index_included'] = int(sample_select[0])
        elif len(sample_select) == 2 and sample_select[1] != '':
            parameter_dict['index_start'] = int(sample_select[0])
            parameter_dict['index_included'] = int(sample_select[1])
        else:
            print('[!] Error parameter with 401 error code.')
            exit(401)
    if not os.path.exists('json/'):
        os.makedirs('json/')
    parameter_json = dict_to_json(parameter_dict, write_file=True, file_name='json/parameter_' + name + '.json')
    print(parameter_json)

    # gpu processing, for further set
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.475, allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        model = Unet3D(sess=sess, parameter_dict=parameter_dict)
        if parameter_dict['phase'] == 'train':
            print('Training Phase...')
            model.train()
        if parameter_dict['phase'] == 'test':
            print('Testing Phase...')
            model.test()


if __name__ == '__main__':
    tf.app.run()
