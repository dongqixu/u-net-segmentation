import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nibabel as nib
import numpy as np


class Visualization:
    def __init__(self, path):
        image = nib.load(path).get_data().astype('uint8')
        image = image * 120
        dimension = np.asarray(image.shape)
        print(image.dtype, np.min(image), np.max(image))
        print(dimension)

        self.image = image
        self.dimension = dimension
        self.index = 0
        self.im = None

    def get_slice(self, index):
        return self.image[:, :, int(index)]

    def update_fig(self, *args):
        self.index += 1
        if self.index >= self.dimension[2]:
            self.index = 0
        self.im.set_array(self.get_slice(self.index))
        return self.im

    def visualize(self):
        fig = plt.figure()
        self.im = plt.imshow(self.get_slice(self.index), animated=True)  # cmap='magma'
        ani = animation.FuncAnimation(fig, self.update_fig, interval=50)
        plt.show()

    # # rotate
    # rotate_image = rotate(image, angle=25, axes=(1, 0), reshape=False, order=1)
    # rotate_dimension = np.asarray(rotate_image.shape)
    #
    # for k in range(rotate_dimension[2]):
    #     cv2.imshow('Rotate_Slice', np.concatenate((image[:, :, k], rotate_image[:, :, k]), axis=1))
    #     cv2.waitKey(0)


if __name__ == '__main__':
    # file = '../hvsmr/crop/data/training_axial_crop_pat0.nii.gz'
    file = '../hvsmr/crop/label/training_axial_crop_pat0-label.nii.gz'
    vi = Visualization(file)
    vi.visualize()
