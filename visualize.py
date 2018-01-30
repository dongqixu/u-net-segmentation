import nibabel as nib
import numpy as np
import cv2
from scipy.ndimage import rotate


def visualize(path):
    image = nib.load(path).get_data().astype('uint8')
    dimension = np.asarray(image.shape)
    image = image * 255
    print(image.dtype, np.max(image), np.min(image))
    print(dimension)

    for k in range(dimension[2]):
        cv2.imshow('Slice', image[:, :, k])
        cv2.waitKey(0)
    return

    # rotate
    rotate_image = rotate(image, angle=25, axes=(1, 0), reshape=False, order=1)
    rotate_dimension = np.asarray(rotate_image.shape)

    for k in range(rotate_dimension[2]):
        cv2.imshow('Rotate_Slice', np.concatenate((image[:, :, k], rotate_image[:, :, k]), axis=1))
        cv2.waitKey(0)


if __name__ == '__main__':
    # file = '../hvsmr/crop/data/training_axial_crop_pat0.nii.gz'
    file = '../hvsmr/crop/label/training_axial_crop_pat0-label.nii.gz'
    visualize(file)
