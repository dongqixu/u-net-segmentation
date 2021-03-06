import copy
import nibabel as nib
import numpy as np
import sys
import time
from scipy.ndimage import rotate
from skimage.transform import resize

''' Data loading and IO '''


# load all data into memory
def load_image_and_label(image_filelist, label_filelist, resize_coefficient=1):
    image_data_list = []
    label_data_list = []
    for ith_file in range(len(label_filelist)):
        # image
        image_data = nib.load(image_filelist[ith_file]).get_data().copy()
        mean_num = np.mean(image_data)
        deviation_num = np.std(image_data)
        image_data = (image_data - mean_num) / (deviation_num + 1e-5)

        # label
        label_data = nib.load(label_filelist[ith_file]).get_data().copy()
        # check
        if image_data.shape != label_data.shape:
            print('Error with shape mismatch!')
            exit(1)

        # TODO: other data augmentation
        if resize_coefficient != 1:
            # skimage.transform.resize: Resize image to match a certain size.
            print('Perform images resizing...')
            resize_dimension = (np.array(image_data.shape) * resize_coefficient).astype(dtype='int32')
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


# generate batch, only for train phase
def get_image_and_label_batch(image_data_list, label_data_list, input_size, batch_size,
                              channel=1, flip_flag=False, rotation_flag=False):
    image_batch = np.zeros([batch_size, input_size, input_size, input_size, channel], dtype='float32')
    label_batch = np.zeros([batch_size, input_size, input_size, input_size], dtype='int32')

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
        label_temp = None
        image_temp = None
        pass_flag = False
        while not pass_flag:
            # random all possible selection
            depth_select = np.random.randint(depth - input_size + 1)
            height_select = np.random.randint(height - input_size + 1)
            width_select = np.random.randint(width - input_size + 1)

            # cropping
            crop_position = np.array([depth_select, height_select, width_select])
            label_temp = random_label[
                         crop_position[0]:crop_position[0] + input_size,
                         crop_position[1]:crop_position[1] + input_size,
                         crop_position[2]:crop_position[2] + input_size
                         ]
            # 0,1,2,3,4 -> pass
            # TODO: throw away part of defected training data?
            label_set = set(np.unique(label_temp))
            if label_set == {0} and np.random.randint(100) >= 0:
                # print('*', end='')
                continue
            elif len(label_set) == 2 and np.random.randint(100) >= 50:
                # print('!', end='')
                continue
            else:
                pass_flag = True
            image_temp = random_image[
                         crop_position[0]:crop_position[0] + input_size,
                         crop_position[1]:crop_position[1] + input_size,
                         crop_position[2]:crop_position[2] + input_size
                         ]

        # data augmentation with rotation and flipping
        if np.random.random() > 0.333:
            if np.random.random() > 0.5:
                if rotation_flag:
                    # print('Rotating batch...')
                    rotate_angle_list = [90, 180, 270]
                    axes_list = [(0, 1), (0, 2), (1, 2)]
                    _angle = rotate_angle_list[np.random.randint(3)]
                    _axes = axes_list[np.random.randint(3)]
                    image_temp = rotate(input=image_temp, angle=_angle, axes=_axes, reshape=False, order=1)
                    label_temp = rotate(input=label_temp, angle=_angle, axes=_axes, reshape=False, order=0)
            else:
                if flip_flag:
                    # print('Flipping batch...')
                    _axis = np.random.randint(3)
                    image_temp = np.flip(image_temp, axis=_axis)
                    label_temp = np.flip(label_temp, axis=_axis)

        # NDHWC
        image_batch[ith_batch, :, :, :, 0] = image_temp
        label_batch[ith_batch, :, :, :] = label_temp

    return image_batch, label_batch


def slice_visualization(image_data, label_data, batch=False, show_depth=None, show_height=None, show_width=None):
    # batch reshape for non raw data
    if batch:
        batch_size, depth, height, width, channel = image_data.shape
        image_data = np.reshape(image_data, [depth, height, width])
        batch_size, depth, height, width = label_data.shape
        label_data = np.reshape(label_data, [depth, height, width])
    # check shape
    if image_data.shape != label_data.shape:
        print('Dimension mismatch!')
        return
    print(f'Data Dimension: {image_data.shape}')
    depth, height, width = image_data.shape
    if show_depth is None:
        show_depth = np.random.randint(depth)
    if show_height is None:
        show_height = np.random.randint(height)
    if show_width is None:
        show_width = np.random.randint(width)

    data_slice = []
    label_slice = []
    # data
    data_slice.append(image_data[show_depth, :, :])
    data_slice.append(image_data[:, show_height, :])
    data_slice.append(image_data[:, :, show_width])
    # label
    label_slice.append(label_data[show_depth, :, :])
    label_slice.append(label_data[:, show_height, :])
    label_slice.append(label_data[:, :, show_width])

    plt.figure()
    for ith_slice in range(3):
        # print(data_slice[ith_slice].shape)
        # print(label_slice[ith_slice].shape)
        plt.subplot(230+ith_slice+1)
        plt.imshow(data_slice[ith_slice], cmap='gray', origin='lower')
        plt.subplot(230+ith_slice+4)
        plt.imshow(label_slice[ith_slice], cmap='gray', origin='lower')
    # plt.subplot_tool()
    plt.show()
    # plt.text(0.5, 0.5, i, fontsize=12)
    # plt.pause(0.01)


if __name__ == '__main__':
    # pyplot module
    if 'matplotlib.pyplot' not in sys.modules:
        import matplotlib.pyplot as plt

    # Not heavy load of main memory
    from glob import glob
    date_form = 'crop'
    image_list = glob(pathname='{}/*.nii.gz'.format(f'../hvsmr/{date_form}/data'))
    label_list = glob(pathname='{}/*.nii.gz'.format(f'../hvsmr/{date_form}/label'))
    image_list.sort()
    label_list.sort()
    print(image_list)
    print(label_list)
    # load test
    image_data_list, label_data_list = load_image_and_label(image_list, label_list, resize_coefficient=1)
    print('images loaded...')

    # print data shape
    for i in range(len(image_data_list)):
        print(image_data_list[i].shape, label_data_list[i].shape)
    #     print(f'Batch {i}:', end='')
    #     print(np.amin(image_data_list[i]), np.amax(image_data_list[i]), end='')
    #     print(np.amin(label_data_list[i]), np.amax(label_data_list[i]))

    # Testing batch
    for i in range(1000):
        start_time = time.time()
        image_batch, label_batch = get_image_and_label_batch(image_data_list, label_data_list, input_size=64,
                                                             batch_size=1, channel=1, rotation_flag=True,
                                                             flip_flag=True)
        # print(f'Loading time: {time.time() - start_time}')
        # print('data: ', end='')
        # print(image_batch.shape)
        # print('label ', end='')
        # print(label_batch.shape)
        # slice_visualization(image_batch, label_batch, batch=True)
    slice_visualization(image_data_list[0], label_data_list[0])
