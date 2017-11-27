import numpy as np


def get_dice_loss(a, b):
    pre = np.array([a, b], dtype=float)
    label = np.array([0, 0], dtype=float)
    inter = np.sum(pre * label)
    union = np.sum(pre * pre + label * label)
    return 1 - 2 * inter / union


def get_weight_list(ratio):
    ratio_x, ratio_y, ratio_z = ratio
    weight_x = 1 - ratio_x
    weight_y = ratio_x / (ratio_y + 1e-5)
    weight_z = ratio_x / (ratio_z + 1e-5)
    weight_y = np.log10(weight_y) + (1 - ratio_y) * 0.5
    weight_z = np.log10(weight_z) + (1 - ratio_z) * 0.5
    weight_sum = weight_x + weight_y + weight_z
    weight_x, weight_y, weight_z = \
        2 * weight_x / weight_sum, 2 * weight_y / weight_sum, 2 * weight_z / weight_sum
    return weight_x, weight_y, weight_z


if __name__ == '__main__':
    for i in range(11):
        i = i / 10
        print(f'{0}->{i}: {get_dice_loss(i, 0.5)}')
    print('------')
    for i in range(11):
        i = i / 10
        print(f'{1}->{i}: {get_dice_loss(0.5, i)}')
    print(get_weight_list((0.786750573, 0.043663444, 0.169585983)))
    print(get_weight_list((1, 0, 0)))
