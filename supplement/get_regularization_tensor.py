import re
from json_io import dict_to_json, json_to_dict


def extract_tensor_name(file_read='name_list.txt', file_write='../regularization_dilated.json', flag=False):
    tensor_dict = dict()
    count = 0
    with open(file_read) as name:
        for line in name:
            line = line.strip().split('\'')[1]
            if re.match('.*batch_norm.*', line):
                pass
            elif re.match('.*bias.*', line):
                pass
            else:
                tensor_dict[count] = line
                count += 1
    json_string = dict_to_json(tensor_dict, write_file=True, file_name=file_write)
    if flag:
        print(json_string)
        _dict = json_to_dict(file_write, read_file=True)
        for element in _dict.values():
            print(element)


if __name__ == '__main__':
    extract_tensor_name()
