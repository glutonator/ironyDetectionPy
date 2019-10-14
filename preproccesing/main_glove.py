# from nltk.tokenize import sent_tokenize, word_tokenize

from pandas import DataFrame

from preproccesing.load_files import load_glove_and_fastText_model, load_input_data, save_output_data, load_FastText_model
from preproccesing.preprocesing import clean_messages, tokenize_data, translate_sentence_to_vectors, create_encoders
import datetime

# embeddingsPath = 'embeddings/'
# input_filesPath = 'input_files/'
# preprocessed_dataPath = 'preprocessed_data/'
# vector_dataPath = 'vector_data/'
#
# model_file = 'word2vec.txt'
# # model_file = 'wiki-news-300d-1M.vec'
# input_file = 'SemEval2018-T3-train-taskA_merged_with_gold_test_taskA.txt'
# preprocessed_file = 'preprocessed_data_fastText_dataset_one.txt'

def preprocess_data():
    # wczytywanie modelu z plliku:
    model = load_glove_and_fastText_model(embeddingsPath + model_file)
    # data: DataFrame = load_input_data('SemEval2018-T3-train-taskA.txt')
    data: DataFrame = load_input_data(input_filesPath+input_file)
    #
    clean_messages(data, model)
    # save to file
    # save_output_data(data, 'preprocessed_data_new2.txt')
    save_output_data(data, preprocessed_dataPath + preprocessed_file)


def prepare_data_for_network():
    # model = load_glove_model('word2vec_50.txt')
    model = load_glove_and_fastText_model('word2vec_200.txt')
    print("model loaded")
    # data: DataFrame = load_input_data('preprocessed_data_new2.txt')
    data: DataFrame = load_input_data('preprocessed_data_new3_merged.txt')
    # data: DataFrame = load_input_data('preprocessed_data_without_blanck_rows.txt')
    # data: DataFrame = load_input_data('preprocessed_data_without_611.txt')
    # data: DataFrame = load_input_data('preprocessed_data.txt')
    # data: DataFrame = load_input_data('xxx.txt')
    print("data loaded")
    tokenize_data(data)
    print("tokenize_data finished")

    label_encoder, onehot_encoder = create_encoders()
    # list_of_not_found_words = \
    #     translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_50.txt',
    #                                   label_encoder=label_encoder, onehot_encoder=onehot_encoder)
    list_of_not_found_words = \
        translate_sentence_to_vectors(data, model, output_filename='vector_test_new_glove_merged_200.txt',
                                      label_encoder=label_encoder, onehot_encoder=onehot_encoder)

    print("translate_sentence_to_vectors finished")
    print("_________________________________")
    print(list_of_not_found_words)
    print("size:" + str(len(list_of_not_found_words)))



start = datetime.datetime.now()

preprocess_data()
# ft_preprocess_data()
prepare_data_for_network()
# ft_prepare_data_for_network()


stop = datetime.datetime.now()
delta = stop - start
print(delta)


