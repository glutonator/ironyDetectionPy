import datetime
import os
import time

import keras
from keras import Sequential
from typing import List
import detection.irony_models_gpu as di_gpu

from detection.my_plots import generate_plots


def train_model_learing_rate(model: Sequential, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, lr):
    gen_report(model, path, "report.txt")
    file_with_model_weights = "weights.best.hdf5"

    model.summary()
    adam = keras.optimizers.Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    # min loss
    # save_best = keras.callbacks.ModelCheckpoint(path + file_with_model_weights, monitor='val_loss', verbose=1,
    #                                             save_best_only=True)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    #

    # max acc
    save_best = keras.callbacks.ModelCheckpoint(path + file_with_model_weights, monitor='val_acc', verbose=1,
                                                save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1)

    cb = [early_stop, save_best]
    # cb = [save_best]
    results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                        # callbacks=cb, epochs=100, batch_size=20,
                        callbacks=cb, epochs=1, batch_size=2000,
                        verbose=0)
    # verbose=0)
    # todo: change to old value of epochs = 100
    # results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
    #                     callbacks=cb, epochs=10, batch_size=5,
    #                     verbose=0)

    generate_plots(results, path)
    scores = model.evaluate(X_test, Y_test, verbose=1)
    scores2222 = model.evaluate(X_val, Y_val, verbose=1)
    save_scores_to_file(model.metrics_names, scores, path, "test_last_scores_testing.txt")
    save_scores_to_file(model.metrics_names, scores2222, path, "test_last_scores_validation.txt")
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
    save_scores_to_file(model.metrics_names, scores, path, "test_best_scores_testing.txt")
    # print(model.metrics_names)
    # print(scores)


def eval_model_validation(model, X_test, Y_test, path):
    # load weights
    file_with_model_weights = "weights.best.hdf5"
    model.load_weights(path + file_with_model_weights)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    scores = model.evaluate(X_test, Y_test, verbose=1)
    save_scores_to_file(model.metrics_names, scores, path, "test_best_scores_validation.txt")
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


# def get_all_models(total_length, max_sentence_length):
#     models: List[Sequential] = []
#
#     # models.append(di_gpu.give_model_10(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_20(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_30(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_40(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_50(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_60(total_length, max_sentence_length))
#     #
#     # models.append(di_gpu.give_model_41(total_length, max_sentence_length))
#     # models.append(di_gpu.give_model_61(total_length, max_sentence_length))
#     #
#     # models.append(di_gpu.give_model_50000(total_length, max_sentence_length))
#     models.append(di_gpu.give_model_50001(total_length, max_sentence_length))
#
#     return models


def trail_all(models: List[Sequential], global_path_to_results, total_length, max_sentence_length, X_train, X_val,
              X_test, Y_train, Y_val, Y_test):
    ts = str(round(time.time()))
    # di_gpu.baseline_00(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # models: List[Sequential] = get_all_models(total_length, max_sentence_length)

    start = datetime.datetime.now()

    for model, function_name in models:
        path = global_path_to_results + ts + "_" + function_name + "/"
        create_dir(path)
        # di.train_model(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path)
        train_model_learing_rate(model, X_train, X_val, X_test, Y_train, Y_val, Y_test, path, 0.0001)
        eval_model(model, X_test, Y_test, path)
        eval_model_validation(model, X_val, Y_val, path)
        print(function_name)

    stop = datetime.datetime.now()
    delta = stop - start
    print(delta)
