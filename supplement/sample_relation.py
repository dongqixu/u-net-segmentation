import matplotlib.pyplot as plt


def plot(scale=False, **evaluation):

    dice_1 = [0.533200832445988, 0.7820184925273768, 0.7943603997936961,
              0.7910531491552513, 0.828554449140995, 0.858750790775064]
    dice_2 = [0.7720918598275812, 0.8590161920387657, 0.8478070711283161,
              0.8832301381181997, 0.8936920599644647, 0.8969054649798416]

    dice_3 = [0.6797871813675562, 0.7576679564766066, 0.8017566476968181,
              0.8080119766377187, 0.8270830139449796, 0.8422100195084155]
    dice_4 = [0.8425565426279472, 0.8794702219681975, 0.8658229727786961,
              0.8848623920292338, 0.9053833982120483, 0.9020014795695249]

    save_file = 'dice'
    x = 'Training Sample'
    y = 'Dice'

    train_epoch = 6
    x_axis = [_i+1 for _i in range(train_epoch)]

    plt.plot(x_axis, dice_2, label='Blood Pool 1')
    plt.plot(x_axis, dice_4, label='Blood Pool 2')
    plt.plot(x_axis, dice_1, label='Myocardium 1')
    plt.plot(x_axis, dice_3, label='Myocardium 2')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle('')
    plt.ylim((0.35, 1))
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{save_file}.jpg')
    plt.show()


def plot2(scale=False, **evaluation):

    jaccard_1 = [0.36912481799746144, 0.6468005206810961, 0.6629976284055026,
                 0.6599244160340126, 0.7053209799210304, 0.7536135226715239]

    jaccard_2 = [0.630942736518253, 0.7533934361081998, 0.7362751230720034,
                 0.7910419904111492, 0.8150834006624554, 0.813906577393425]

    jaccard_3 = [0.5371589588102793, 0.6287624889107194, 0.6530519900842904,
                 0.684102072168424, 0.7121416365992375, 0.7303281583075227]
    jaccard_4 = [0.7292438994867861, 0.7849141202699692, 0.7903210972651493,
                 0.7940364305889385, 0.827410875057153, 0.82202842772896]

    save_file = 'jaccard'
    x = 'Training Sample'
    y = 'Jaccard'

    train_epoch = 6
    x_axis = [_i+1 for _i in range(train_epoch)]

    plt.plot(x_axis, jaccard_2, label='Blood Pool 1')
    plt.plot(x_axis, jaccard_4, label='Blood Pool 2')
    plt.plot(x_axis, jaccard_1, label='Myocardium 1')
    plt.plot(x_axis, jaccard_3, label='Myocardium 2')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle('')
    plt.ylim((0.35, 1))
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{save_file}.jpg')
    plt.show()


def plot3(scale=False, **evaluation):

    dice_1 = [0.635760, 0.748780, 0.774607, 0.809653, 0.834784, 0.843303]
    dice_2 = [0.810210, 0.844003, 0.861574, 0.886169, 0.892515, 0.895720]
    increase_1 = [0.0]
    increase_2 = [0.0]
    for i in range(5):
        increase_1.append((dice_1[i+1] - dice_1[i]) * 5)
        increase_2.append((dice_2[i + 1] - dice_2[i]) * 5)
    # jaccard_1 = [0.482482, 0.610840, 0.635909, 0.686347, 0.719245, 0.732437]
    # jaccard_2 = [0.688844, 0.733942, 0.766637, 0.796276, 0.807562, 0.811465]

    save_file = 'jaccard_dice'
    x = 'Training Sample'
    y = 'Dice'

    train_epoch = 6
    x_axis = [_i+1 for _i in range(train_epoch)]

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    plt.plot(x_axis, dice_2, label='Blood Pool Dice')
    plt.plot(x_axis, dice_1, label='Myocardium Dice')
    plt.plot(x_axis, increase_2, label='Blood Pool Increase')
    plt.plot(x_axis, increase_1, label='Myocardium Increase')
    # plt.plot(x_axis, jaccard_1, label='Blood Pool Jaccard')
    # plt.plot(x_axis, jaccard_2, label='Myocardium Jaccard')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle('')
    plt.ylim((0, 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_file}.jpg')
    plt.show()


def plot4(scale=False, **evaluation):
    jaccard_1 = [0.482482, 0.610840, 0.635909, 0.686347, 0.719245, 0.732437]
    jaccard_2 = [0.688844, 0.733942, 0.766637, 0.796276, 0.807562, 0.811465]
    increase_1 = [0.0]
    increase_2 = [0.0]
    for i in range(5):
        increase_1.append((jaccard_1[i+1] - jaccard_1[i]) * 5)
        increase_2.append((jaccard_2[i + 1] - jaccard_2[i]) * 5)

    save_file = 'jaccard_dice_j'
    x = 'Training Sample'
    y = 'Jaccard'

    train_epoch = 6
    x_axis = [_i+1 for _i in range(train_epoch)]

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    plt.plot(x_axis, jaccard_2, label='Blood Pool Jaccard')
    plt.plot(x_axis, jaccard_1, label='Myocardium Jaccard')
    plt.plot(x_axis, increase_2, label='Blood Pool Increase')
    plt.plot(x_axis, increase_1, label='Myocardium Increase')

    plt.xlabel(x)
    plt.ylabel(y)
    plt.suptitle('')
    plt.ylim((0, 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_file}.jpg')
    plt.show()


if __name__ == '__main__':
    plot()
    plot2()
    plot3()
    plot4()
