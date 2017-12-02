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
    parameter_dict['phase'] = 'train'
    parameter_dict['batch_size'] = 1
    parameter_dict['input_size'] = 64
    parameter_dict['input_channels'] = 1
    parameter_dict['output_size'] = 64
    parameter_dict['output_channels'] = 3
    parameter_dict['learning_rate'] = 0.001
    parameter_dict['beta1'] = 0.5
    parameter_dict['epoch'] = 35000  # 100000 -> 10000
    date_form = 'crop'
    parameter_dict['train_data_dir'] = f'../hvsmr/{date_form}/data/'
    parameter_dict['test_data_dir'] = f'../hvsmr/{date_form}/data/'
    parameter_dict['label_data_dir'] = f'../hvsmr/{date_form}/label/'
    parameter_dict['prediction_dir'] = f'../hvsmr/{date_form}/prediction/'
    parameter_dict['model_name'] = f'hvsmr_{date_form}_{name}.model'
    parameter_dict['name_with_runtime'] = name
    parameter_dict['checkpoint_dir'] = 'checkpoint/'
    parameter_dict['resize_coefficient'] = 1
    parameter_dict['test_stride'] = 32  # for overlap
    # from previous version
    parameter_dict['save_interval'] = 5000  # 10000 -> 1000
    parameter_dict['test_interval'] = 1000  # ResNet
    parameter_dict['cube_overlapping_factor'] = 4
    parameter_dict['gpu'] = '0'

    # scalable number of feature maps: default 32
    parameter_dict['feature_number'] = 16  # 32 -> 16
    parameter_dict['index_start'] = 0
    parameter_dict['index_included'] = 4

    # for experiment
    parameter_dict['rotation'] = True
    parameter_dict['dice_option'] = 'value'
    parameter_dict['regularization'] = False
    parameter_dict['network'] = 'unet'
    parameter_dict['log_weight'] = False
    parameter_dict['dice_loss_coefficient'] = 0.25  # 0.5 -> 0.25
    parameter_dict['l2_coefficient'] = 0.0005
    parameter_dict['select_sample'] = None

    return parameter_dict


# What is the input parameter
def main(_):
    parser = argparse.ArgumentParser(description='Process argument for parameter dictionary.')
    parser.add_argument('-g', '--gpu', help='cuda visible devices')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--sample', help='sample selection')
    # experiment
    parser.add_argument('-R', '--rotation', action='store_true')
    parser.add_argument('-d', '--dice', choices=['value', 'softmax'], help='dice type')
    parser.add_argument('-r', '--regularization', action='store_true')
    parser.add_argument('-n', '--network', choices=['unet', 'dilated'], help='network option')
    parser.add_argument('--epoch', help='training epochs')
    parser.add_argument('--save_interval', help='save interval')
    parser.add_argument('--test_interval', help='test interval')
    parser.add_argument('--memory', help='memory usage for unlimited usage')
    parser.add_argument('--log_weight', action='store_true')
    parser.add_argument('--dice_coefficient', help='multiple of dice loss')
    parser.add_argument('--l2_coefficient', help='multiple of l2 loss')
    parser.add_argument('--select', help='select samples from list')
    # parser.add_argument('--feature', help='number of features')
    # TODO:
    args = parser.parse_args()
    if args.gpu:
        gpu = args.gpu
    else:
        gpu = '0'

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
    if args.rotation:
        parameter_dict['rotation'] = True
    if args.dice:
        parameter_dict['dice_option'] = args.dice
    if args.regularization:
        parameter_dict['regularization'] = True
    if args.network:
        parameter_dict['network'] = args.network
    if args.epoch:
        parameter_dict['epoch'] = int(args.epoch)
    if args.save_interval:
        parameter_dict['save_interval'] = int(args.save_interval)
    if args.test_interval:
        parameter_dict['test_interval'] = int(args.test_interval)
    if args.log_weight:
        parameter_dict['log_weight'] = True
    if args.dice_coefficient:
        parameter_dict['dice_loss_coefficient'] = float(args.dice_coefficient)
    if args.l2_coefficient:
        parameter_dict['l2_coefficient'] = float(args.l2_coefficient)
    if args.select:
        sample = list()
        string = args.select.strip().split(',')
        for var in string:
            sample.append(int(var))
        parameter_dict['select_sample'] = sample

    if not os.path.exists('json/'):
        os.makedirs('json/')
    parameter_json = dict_to_json(parameter_dict, write_file=True, file_name='json/parameter_' + name + '.json')
    print(parameter_json)

    # gpu processing, for further set
    if args.memory:
        memory = float(args.memory)
    else:
        memory = 0.475
    print(f'Memory fraction: {memory}')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory, allow_growth=True)

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
