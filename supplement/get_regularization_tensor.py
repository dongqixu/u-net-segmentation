import re
from json_io import dict_to_json, json_to_dict


def extract_tensor_name():
    tensor_dict = dict()
    count = 0
    with open('name_list.txt') as name:
        for line in name:
            line = line.strip().split('\'')[1]
            if re.match('.*batch_norm.*', line):
                pass
            elif re.match('.*bias.*', line):
                pass
            else:
                tensor_dict[count] = line
                count += 1
    json_string = dict_to_json(tensor_dict, write_file=True, file_name='../regularization.json')
    print(json_string)

    _dict = json_to_dict('../regularization.json', read_file=True)
    for element in _dict.values():
        print(element)


if __name__ == '__main__':
    extract_tensor_name()
