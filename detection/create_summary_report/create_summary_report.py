import os
from typing import List, Dict
from typing import Tuple
import pprint

global_path = '../../results2_ft_merged/' + 'dataset_three__balanced_figurative_eval_one/'
path = '../../results2_ft_merged/' + 'dataset_three__balanced_figurative_eval_one/' + '1573398849_give_model_00/'

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
    label: str = line[0]
    value: str = line[1]
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


directory = '../../results2_ft_merged/' + 'dataset_three__balanced_figurative_eval_one/'


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




def get_dict_with_metrics(directory: str, main_name='1573398849_') -> Dict:
    list_of_subdir: List[str] = get_list_of_subdirectories(directory)
    list_of_model_names: List[str] = []

    dict_files = {}

    for sub_dir_name in list_of_subdir:
        model_name: str = get_model_name(sub_dir_name, main_name)

        local_path = global_path  + sub_dir_name + "/"
        file = 'one_test_best_scores_other_metrics.txt'
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

tmp_dict = get_dict_with_metrics(directory)
print(convert_dict_for_save_to_csv(tmp_dict))

