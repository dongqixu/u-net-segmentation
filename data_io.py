import copy
import nibabel as nib
import numpy as np
from skimage.transform import resize

''' Data loading and IO '''


# load all data into memory
def load_image_and_label(image_filelist, label_filelist, resize_coefficient=1):
    image_data_list = []
    label_data_list = []
    for ith_file in range(len(label_filelist)):
        # image
        image_data = nib.load(image_filelist[ith_file]).get_data()
        image_data = copy.copy(image_data)
        # label
        label_data = nib.load(label_filelist[ith_file]).get_data()
        label_data = copy.copy(label_data)
        # check
        if image_data.shape != label_data.shape:
            print('Error with shape mismatch!')
            exit(1)

        # TODO: other data argumentation
        '''Resize or other method'''
        if resize_coefficient != 1:
            # skimage.transform.resize: Resize image to match a certain size.
            print('Perform images resizing...')
            resize_dimension = (np.array(image_data.shape) * resize_coefficient).astype(dtype='int')
            image_data = resize(image=image_data, output_shape=resize_dimension, order=1,
                                preserve_range=True, mode='constant')
            label_data = resize(image=label_data, output_shape=resize_dimension, order=0,
                                preserve_range=True, mode='constant')
            # The order of interpolation. The order has to be in the range 0-5:
            # 0: Nearest-neighbor
            # 1: Bi-linear (default)
            # 2: Bi-quadratic
            # 3: Bi-cubic
            # 4: Bi-quartic
            # 5: Bi-quintic
            # preserve_range: keep the original range of values
        image_data_list.append(image_data)
        label_data_list.append(label_data)

    return image_data_list, label_data_list


# generate batch
def get_image_and_label_batch(image_data_list, label_data_list, input_size, batch_size,
                              channel=1, flip_flag=False, rotation_flag=False):
    image_batch = np.zeros([batch_size, input_size, input_size, input_size, channel]).astype('float32')
    label_batch = np.zeros([batch_size, input_size, input_size, input_size]).astype('int32')

    '''Cannot make sure all data go through the network'''
    # random reading data batch
    for ith_batch in range(batch_size):
        # randomly select image file
        random_range = np.arange(len(image_data_list))
        np.random.shuffle(random_range)
        random_image = image_data_list[random_range[0]].astype('float32')
        random_label = label_data_list[random_range[0]].astype('int32')

        # randomly select cube -> boundary considered
        depth, height, width = random_image.shape
        depth_random = np.arange(depth - input_size)
        np.random.shuffle(depth_random)
        height_random = np.arange(height - input_size)
        np.random.shuffle(height_random)
        width_random = np.arange(width - input_size)
        np.random.shuffle(width_random)

        # cropping
        crop_position = np.array([depth_random[0], height_random[0], width_random[0]])
        image_temp = copy.deepcopy(
            random_image[
                crop_position[0]:crop_position[0] + input_size,
                crop_position[1]:crop_position[1] + input_size,
                crop_position[2]:crop_position[2] + input_size
            ]
        )
        label_temp = copy.deepcopy(
            random_label[
                crop_position[0]:crop_position[0] + input_size,
                crop_position[1]:crop_position[1] + input_size,
                crop_position[2]:crop_position[2] + input_size
            ]
        )

        '''Necessary? Zero problem may happen'''
        # TODO: normalization over the whole dataset?
        image_temp = image_temp / 255.0  # what is the range?
        mean_temp = np.mean(image_temp)
        deviation_temp = np.std(image_temp)
        # print(mean_temp, deviation_temp)  # the form of single value
        image_normalization = (image_temp - mean_temp) / (deviation_temp + 1e-5)

        # data augmentation with rotation
        # TODO: data augmentation
        if rotation_flag and np.random.random() > 0.65:
            print('Rotation batch...')
            '''Further improvement on rotation'''
        if flip_flag and np.random.random() > 0.65:
            print('Flipping batch...')
            '''Further improvement on flipping'''

        # NDHWC
        image_batch[ith_batch, :, :, :, channel-1] = image_normalization
        label_batch[ith_batch, :, :, :] = label_temp

    return image_batch, label_batch


if __name__ == '__main__':
    # Not heavy load of main memory
    from glob import glob
    image_list = glob(pathname='{}/*.nii.gz'.format('../hvsmr/data'))
    label_list = glob(pathname='{}/*.nii.gz'.format('../hvsmr/label'))
    image_list.sort()
    label_list.sort()
    print(image_list)
    print(label_list)
    # load test
    image_data_list, label_data_list = load_image_and_label(image_list, label_list, resize_coefficient=1)
    print('images loaded...')

    # print shape
    for i in range(len(image_data_list)):
        print(f'Batch {i}')
        print(np.amin(image_data_list[i]), np.amax(image_data_list[i]))
        print(np.amin(label_data_list[i]), np.amax(label_data_list[i]))
        print('-----------------------')

    # Testing batch
    for i in range(1000):
        image_batch, label_batch = get_image_and_label_batch(image_data_list, label_data_list, input_size=96,
                                                             batch_size=1, channel=1)
        # print('data')
        # print(image_batch.shape)
        # print('label')
        # print(label_batch.shape)
    # TODO: visualization
