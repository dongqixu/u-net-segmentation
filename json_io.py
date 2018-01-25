import json

''' Json for Parameter Setting and IO '''


# convert dictionary to json
def dict_to_json(dict_data, write_file=False, file_name=None):
    json_str = None
    if write_file and file_name is not None:
        with open(file_name, 'w') as json_write:
            json.dump(dict_data, json_write, indent=4)  # return None
            json_str = json.dumps(dict_data, indent=4)
    elif write_file and file_name is None:
        print('Error without output file name provided.')
    else:
        json_str = json.dumps(dict_data, indent=4)
    return json_str


# convert json to dictionary
def json_to_dict(json_str, read_file=False):
    if read_file:
        with open(json_str, 'r') as json_read:
            dict_data = json.load(json_read)
    else:
        dict_data = json.loads(json_str)
    return dict_data


# convert necessary parameter into json
# sample: self.phase = parameter_dict['phase'] -> "phase": "",

# deprecated
def extract_json_format_from_class_init(text_in_file):
    output_string = '{\n'
    with open(text_in_file, 'r') as text_read:
        for line in text_read:
            line = line.strip().split('\'')
            if len(line) == 3:
                append_string = "    \"%s\": \"\",\n" % line[1]
                output_string = output_string + append_string
    output_string = output_string[:-2] + '\n}\n'
    return output_string
