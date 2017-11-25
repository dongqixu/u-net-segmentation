import numpy as np


def get_dice_loss(a, b):
    pre = np.array([a, b], dtype=float)
    label = np.array([0, 0], dtype=float)
    inter = np.sum(pre * label)
    union = np.sum(pre * pre + label * label)
    return 1 - 2 * inter / union


if __name__ == '__main__':
    for i in range(11):
        i = i / 10
        print(f'{0}->{i}: {get_dice_loss(i, 0.5)}')
    print('------')
    for i in range(11):
        i = i / 10
        print(f'{1}->{i}: {get_dice_loss(0.5, i)}')
