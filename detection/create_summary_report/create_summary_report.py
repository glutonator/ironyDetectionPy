import os
from typing import List, Dict
from typing import Tuple
import pprint
import csv

global_path = '../../results3_ft_merged/' + '3_cnn_test/'
path = '../../results3_ft_merged/' + 'test_run/' + '1573398849_give_model_00/'

file = 'one_test_best_scores_other_metrics.txt'

path_to_file = path + file


def read_from_file(path_to_file: str) -> List[str]:
    with open(path_to_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content: List[str] = [x.strip() for x in content]
    return content


def get_label_with_value(line: str) -> Tuple[str, str]:
    line = line.split(":")
    label: str = line[0].strip()
    value: str = line[1].strip()
    return label, value


def read_values_from_list(content: List[str]) -> Dict:
    dict_of_metrics = {}
    for line in content:
        label: str
        value: str
        label, value = get_label_with_value(line)
        dict_of_metrics[label] = value
    return dict_of_metrics


def create_dict_with_values_from_file(path_to_file: str):
    # local_path = global_path + sub_dir_name + "/"
    # file = 'one_test_best_scores_other_metrics.txt'
    # local_path_to_file = local_path + file
    return read_values_from_list(read_from_file(path_to_file))


directory = '../../results3_ft_merged/' + '3_cnn_test/'


def get_list_of_subdirectories(directory: str) -> List[str]:
    # print ([x[0] for x in os.walk(directory)])
    list_of_subdir = next(os.walk(directory))[1]
    return list_of_subdir


def get_model_name(sub_dir_name: str, main_name: str) -> str:
    return remove_prefix_from_str(sub_dir_name, main_name)


def remove_prefix_from_str(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix):]
    return string


def get_dict_with_metrics(directory: str, main_name='') -> Dict:
    list_of_subdir: List[str] = get_list_of_subdirectories(directory)
    list_of_model_names: List[str] = []

    dict_files = {}

    for sub_dir_name in list_of_subdir:
        model_name: str = get_model_name(sub_dir_name, main_name)

        local_path = global_path + sub_dir_name + "/"
        file = 'test_best_scores_other_metrics.txt'
        local_path_to_file = local_path + file
        dict_of_metrics: Dict = create_dict_with_values_from_file(local_path_to_file)
        # tmp = read_from_file(local_path_to_file)
        dict_files[model_name] = dict_of_metrics

        list_of_model_names.append(model_name)
    # pprint.pprint(dict_files)
    return dict_files


def convert_dict_for_save_to_csv(loc_dict: Dict):
    output_list = []
    value: Dict
    for key, value in loc_dict.items():
        tmp_dict = value.copy()
        tmp_dict['model'] = key
        output_list.append(tmp_dict)

    return output_list


main_name = '1577290625_'
tmp_dict = get_dict_with_metrics(directory, main_name=main_name)
data = convert_dict_for_save_to_csv(tmp_dict)

# data = [{'mountain' : 'Everest', 'height': '8848'},
#       {'mountain' : 'K2 ', 'height': '8611'},
#       {'mountain' : 'Kanchenjunga', 'height': '8586'}]
with open(main_name + 'test_best_scores_other_metrics.csv', 'w') as csvFile:
    # data[0].keys()
    # fields = [ 'height', 'mountain']


    # fields = data[0].keys()
    # accuracy ,precision ,recall ,f1 ,model
    fields = ['model', 'accuracy', 'precision', 'recall', 'f1']
    # fields = ['model']
    writer = csv.DictWriter(csvFile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data)
print("writing completed")
csvFile.close()
