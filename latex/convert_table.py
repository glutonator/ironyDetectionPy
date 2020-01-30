import os
from os import listdir
from os.path import isfile, join

path_in = 'latex_tables_in/'
path_out = 'latex_tables_out/'

def func(filename: str):

    # file = 'Iliad.txt'
    file = filename
    filepath = path_in + file

    with open(filepath) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    flag = False


    def method_name(output_list_per_line, element1: str, elemnt2: str):
        if float(element1) > float(list_of_elemnts[1]):
            output_list_per_line.append("\\textbf{" + element1 + "}")
            output_list_per_line.append(elemnt2)
        else:
            output_list_per_line.append(element1)
            output_list_per_line.append("\\textbf{" + elemnt2 + "}")


    output_list = []
    for line in content:
        if "sieć" in line:
            flag = True
            output_list.append(line)
            continue
        if "tabular" in line:
            flag = False
            output_list.append(line)
            continue
        if flag:
            splited_elements = line.split('&')
            network_name = splited_elements[0]
            list_of_elemnts = []
            merged_other_elements = None
            for element in splited_elements[1:]:
                trimed_element = element.strip()
                if "hline" in trimed_element:
                    splited_elements_for_last = trimed_element.split(" ")
                    last_element = splited_elements_for_last[0]
                    merged_other_elements = " ".join(splited_elements_for_last[1:])
                    list_of_elemnts.append(last_element)
                else:
                    list_of_elemnts.append(trimed_element)

                # print(trimed_element)

            # porownywanie
            output_list_per_line = []
            method_name(output_list_per_line, list_of_elemnts[0], list_of_elemnts[1])
            method_name(output_list_per_line, list_of_elemnts[2], list_of_elemnts[3])
            method_name(output_list_per_line, list_of_elemnts[4], list_of_elemnts[5])
            method_name(output_list_per_line, list_of_elemnts[6], list_of_elemnts[7])
            # print(output_list_per_line)
            # print(" & ".join(output_list_per_line))
            merged = " & ".join(output_list_per_line)
            merged = network_name + " & " + merged + " " + merged_other_elements
            output_list.append(merged)

            # if float(list_of_elemnts[0]) > float(list_of_elemnts[1]):
            #     output_list.append("\\textbf{" + list_of_elemnts[0] + "}")
            #     output_list.append(list_of_elemnts[1])
            # else:
            #     output_list.append(list_of_elemnts[0])
            #     output_list.append("\\textbf{" + list_of_elemnts[1] + "}")
        else:
            output_list.append(line)

    print(output_list)

    # with open('./output.txt', 'w') as f1:
    output_path = path_out + file
    with open(output_path, 'w') as f1:
        for line in output_list:
            f1.write(line + os.linesep)


onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]
print(onlyfiles)
for xx in onlyfiles:
    func(xx)