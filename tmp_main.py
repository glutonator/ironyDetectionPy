# wczytywanie modelu z plliku:
from pandas import DataFrame
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from preproccesing.load_files import load_glove_and_fastText_model, load_input_data
from preproccesing.preprocesing import tokenize_data, translate_sentence_to_vectors

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model = load_glove_and_fastText_model('word2vec.txt')
# data: DataFrame = load_input_data('tmp_file_in.txt')
# #
# clean_messages(data, model)
# # save to file
# save_output_data(data, 'tmp_file_out.txt')
#


data: DataFrame = load_input_data('tmp_file_out.txt')
print("data loaded")
tokenize_data(data)
print("tokenize_data finished")

print(data)


#########################
def create_encoders():
    list_of_pos_tags = []
    with open('pos_tags.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            print(line)
            list_of_pos_tags.append(line)
            # Do something with 'line'

    print(list_of_pos_tags)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list_of_pos_tags)
    print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return label_encoder, onehot_encoder


####################################3

label_encoder, onehot_encoder = create_encoders()

list_of_not_found_words = translate_sentence_to_vectors(data, model, output_filename='tmp_xxxxx.txt',
                                                        label_encoder=label_encoder, onehot_encoder=onehot_encoder)

print("translate_sentence_to_vectors finished")
print("_________________________________")
print(list_of_not_found_words)
print("size:" + str(len(list_of_not_found_words)))

# clean_messages(data, model)
# print(data['Tweet_text'])
