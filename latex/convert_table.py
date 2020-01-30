import os
from os import listdir
from os.path import isfile, join

path_in = 'latex_tables_in/'
path_out = 'latex_tables_out/'


def method_name(output_list_per_line, element1: str, elemnt2: str, max1, max2):
    if float(element1) > float(elemnt2):
        # output_list_per_line.append("\\textbf{" + element1 + "}")
        output_list_per_line.append("\\textbf{" + pverright_value_with_color_for_max(element1, max1) + "}")
        output_list_per_line.append(pverright_value_with_color_for_max(elemnt2, max2))
    else:
        output_list_per_line.append(pverright_value_with_color_for_max(element1, max1))
        output_list_per_line.append("\\textbf{" + pverright_value_with_color_for_max(elemnt2, max2) + "}")


def pverright_value_with_color_for_max(value: str, max: float):
    if float(value) == max:
        value = "\cellcolor{green!50}" + value
        return value
    else:
        return value


def func(filename: str):
    # file = 'Iliad.txt'
    file = filename
    filepath = path_in + file

    with open(filepath) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    flag = False

    # max value
    # mapa_wartosci = {}
    lista_0 = []
    lista_1 = []
    lista_2 = []
    lista_3 = []
    lista_4 = []
    lista_5 = []
    lista_6 = []
    lista_7 = []
    for line in content:
        if "sieć" in line:
            flag = True
            # output_list.append(line)
            continue
        if "tabular" in line:
            flag = False
            # output_list.append(line)
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
            lista_0.append(float(list_of_elemnts[0]))
            lista_1.append(float(list_of_elemnts[1]))
            lista_2.append(float(list_of_elemnts[2]))
            lista_3.append(float(list_of_elemnts[3]))
            lista_4.append(float(list_of_elemnts[4]))
            lista_5.append(float(list_of_elemnts[5]))
            lista_6.append(float(list_of_elemnts[6]))
            lista_7.append(float(list_of_elemnts[7]))
            # method_name(output_list_per_line, list_of_elemnts[0], list_of_elemnts[1])
            # method_name(output_list_per_line, list_of_elemnts[2], list_of_elemnts[3])
            # method_name(output_list_per_line, list_of_elemnts[4], list_of_elemnts[5])
            # method_name(output_list_per_line, list_of_elemnts[6], list_of_elemnts[7])
            # print(output_list_per_line)
            # print(" & ".join(output_list_per_line))
            merged = " & ".join(output_list_per_line)
            merged = network_name + " & " + merged + " " + merged_other_elements
            # output_list.append(merged)

        # else:
        # output_list.append(line)

    najwieksze = [
        max(lista_0),
        max(lista_1),
        max(lista_2),
        max(lista_3),
        max(lista_4),
        max(lista_5),
        max(lista_6),
        max(lista_7)
    ]

    # najwieksze =  [str(i) for i in najwieksze]

    # bolding
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
            method_name(output_list_per_line, list_of_elemnts[0], list_of_elemnts[1], najwieksze[0], najwieksze[1])
            method_name(output_list_per_line, list_of_elemnts[2], list_of_elemnts[3], najwieksze[2], najwieksze[3])
            method_name(output_list_per_line, list_of_elemnts[4], list_of_elemnts[5], najwieksze[4], najwieksze[5])
            method_name(output_list_per_line, list_of_elemnts[6], list_of_elemnts[7], najwieksze[6], najwieksze[7])
            # print(output_list_per_line)
            # print(" & ".join(output_list_per_line))
            merged = " & ".join(output_list_per_line)
            merged = network_name + " & " + merged + " " + merged_other_elements
            output_list.append(merged)

        else:
            output_list.append(line)

    print(output_list)

    # with open('./output.txt', 'w') as f1:
    output_path = path_out + file
    with open(output_path, 'w') as f1:
        for line in output_list:
            f1.write(line + os.linesep)


onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]
onlyfiles.remove('.gitkeep')
print(onlyfiles)
for xx in onlyfiles:
    func(xx)
