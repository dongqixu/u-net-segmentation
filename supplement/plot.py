import re
import matplotlib.pyplot as plt


# extract data from output script
def extract_list(file_name):
    _dice = [[] for _ in range(10)]
    _jaccard = [[] for _ in range(10)]
    _voe = [[] for _ in range(10)]
    _vd = [[] for _ in range(10)]
    _accuracy = [[] for _ in range(10)]
    with open(file_name, 'r') as file:
        epoch = 0
        line = True
        while line:
            line = file.readline()
            string = line.strip()
            if re.match('.*Test.*', string):
                epoch += 1
                # print(f'*** {epoch} ***')
                for ith_sample in range(10):
                    file.readline()  # skip empty line
                    # print(f'=== {ith_sample} ===')
                    file.readline()
                    file.readline()
                    _dice[ith_sample].append(convert_line_to_list(file.readline()))
                    _jaccard[ith_sample].append(convert_line_to_list(file.readline()))
                    _voe[ith_sample].append(convert_line_to_list(file.readline()))
                    _vd[ith_sample].append(convert_line_to_list(file.readline()))
                    _ = file.readline()  # time
                    _accuracy[ith_sample].append(convert_accuracy_to_list(file.readline()))
    return _dice, _jaccard, _voe, _vd, _accuracy


# convert dice, jaccard, voe, vd to data in list form
def convert_line_to_list(line):
    # input line
    line = line.strip().split('[')[2]
    line = line.split(']')[0]
    line = line.split(',')
    # output list
    output = list()
    for element in line:
        output.append(float(element.strip()))
    return output


# convert accuracy to data in list form
def convert_accuracy_to_list(line):
    line = line.strip().replace(',', '')
    line = line.split('[')[3:5+1]
    for ith_class in range(3):
        line[ith_class] = line[ith_class].strip().split(']')[0]
        line[ith_class] = line[ith_class].split(' ')[0]
        line[ith_class] = float(line[ith_class].strip())
    return line


# given specific type of evaluation, compute mean among selected sample
def compute_average(evaluation, scope=(5, 10)):
    # scope = (0, 5) or (5, 10)
    start, stop = scope
    label_0 = list()
    label_1 = list()
    label_2 = list()
    train_epoch = len(evaluation[0])
    for epoch in range(train_epoch):
        average_0 = sum(evaluation[sample][epoch][0] for sample in range(start, stop)) / (stop-start)
        average_1 = sum(evaluation[sample][epoch][1] for sample in range(start, stop)) / (stop-start)
        average_2 = sum(evaluation[sample][epoch][2] for sample in range(start, stop)) / (stop-start)
        label_0.append(average_0)
        label_1.append(average_1)
        label_2.append(average_2)
    return label_0, label_1, label_2


def plot(title, file, scale=False, **evaluation):
    train_epoch = 0
    for key in evaluation:
        value = evaluation[key]
        train_epoch = len(value)
        break
    x_axis = [_i+1 for _i in range(train_epoch)]
    for key in evaluation:
        value = evaluation[key]
        '''display only'''
        if scale:
            for i in range(len(value)):
                value[i] = value[i] * value[i]
        '''display only'''
        plt.plot(x_axis, value, label=key)

        highest = max(value)
        highest_value = [highest for _ in range(len(value))]
        plt.plot(x_axis, highest_value, label=f'{key}_highest')

    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.suptitle(title)
    plt.ylim((0.36, 1))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{file}_.jpg')
    plt.show()


if __name__ == '__main__':
    with open('../test/output/time_order.txt') as _file:
        count = 1
        for _line in _file:
            _line = _line.strip()
            print(f'======== {_line} ========')
            dice, jaccard, voe, vd, accuracy = extract_list(f'../test/output/{_line}')
            dice_0, dice_1, dice_2 = compute_average(jaccard)
            print(f'{max(dice_0)}\t{max(dice_1)}\t{max(dice_2)}')
            plot(_line, count, dice_0=dice_0, dice_1=dice_1, dice_2=dice_2, scale=True)
            count += 1
