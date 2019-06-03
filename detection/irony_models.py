import inspect
import os

import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Flatten

from detection.my_plots import generate_plots


def give_model_00(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(Dense(20, input_shape=(max_sentence_length, len_of_vector_embeddings)))
    # model.add(Dense(40))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_10(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(10, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_20(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_30(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_40(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_50(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(LSTM(10, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(10, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_60(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(LSTM(20, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(20, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


##################################################
# lstm vs bi_lstm
def give_model_41(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_61(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(LSTM(20, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(20, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


####################################################################################
def give_model_X(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_testing(len_of_vector_embeddings, max_sentence_length):
    # path = "results/"
    model: Sequential = Sequential()
    # model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
    # model.add(LSTM(20, input_shape=(11, 25), return_sequences=False))
    # model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    # model.add(LSTM(50, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(
        Bidirectional(LSTM(10, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(10, return_sequences=False)))
    # model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # get function name
    function_name = inspect.currentframe().f_code.co_name

    return model, function_name


def train_model(model: Sequential, X_train, X_val, X_test, Y_train, Y_val, Y_test, path):
    gen_report(model, path, "report.txt")
    file_with_model_weights = "weights.best.hdf5"

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_best = keras.callbacks.ModelCheckpoint(path + file_with_model_weights, monitor='val_loss', verbose=0,
                                                save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)

    # results = model.fit(X_train, Y_train, validation_split=0.2,
    #                     callbacks=[early_stop, save_best], epochs=40, batch_size=10,
    #                     verbose=0)
    results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        callbacks=[early_stop, save_best], epochs=40, batch_size=10,
                        verbose=0)
    generate_plots(results, path)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    save_scores_to_file(model.metrics_names, scores, path, "test_last_scores.txt")
    # print(model.metrics_names)
    # print(scores)


def train_model_learing_rate(model: Sequential, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, lr):
    gen_report(model, path, "report.txt")
    file_with_model_weights = "weights.best.hdf5"

    model.summary()
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    save_best = keras.callbacks.ModelCheckpoint(path + file_with_model_weights, monitor='val_loss', verbose=0,
                                                save_best_only=True)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)

    results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        callbacks=[save_best], epochs=50, batch_size=5,
                        verbose=0)

    generate_plots(results, path)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    save_scores_to_file(model.metrics_names, scores, path, "test_last_scores.txt")
    # print(model.metrics_names)
    # print(scores)


def eval_model(model, X_test, Y_test, path):
    # load weights
    file_with_model_weights = "weights.best.hdf5"
    model.load_weights(path + file_with_model_weights)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    scores = model.evaluate(X_test, Y_test, verbose=1)
    save_scores_to_file(model.metrics_names, scores, path, "test_best_scores.txt")
    # print(model.metrics_names)
    # print(scores)


def create_dir(dir_path):
    # detect the current working directory and print it
    path = os.getcwd()
    print("The current working directory is %s" % path)
    try:
        os.mkdir(path + "/" + dir_path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def gen_report(model: Sequential, path, filename):
    # Open the file
    with open(path + filename, 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def save_scores_to_file(metrics_names, scores, path, filename):
    file = open(path + filename, "w")
    file.write(str(metrics_names) + "\n")
    file.write(str(scores))
    file.close()
