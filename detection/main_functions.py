from __future__ import print_function

import sys

import datetime
import os
import time
import numpy as np

import keras
from keras import Sequential
from typing import List

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                        callbacks=cb, epochs=100, batch_size=5,
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


def eval_f1(model, X_test, Y_test, path):
    # load weights
    file_with_model_weights = "weights.best.hdf5"
    model.load_weights(path + file_with_model_weights)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    # predict probabilities for test set
    yhat_probs = model.predict(X_test, verbose=0)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=0)

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, yhat_classes)
    print('F1 score: %f' % f1)

    params_to_save = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # params_to_save = [accuracy, precision, recall, f1]

    other_metrics_to_file(params_to_save, path, "test_best_scores_other_metrics.txt")


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


# def debug_to_string(expression):
#     frame = sys._getframe(1)
#     return str(expression, '=', repr(eval(expression, frame.f_globals, frame.f_locals)))


def other_metrics_to_file(params_to_save, path, filename):
    file = open(path + filename, "w")
    for key, value in params_to_save.items():
        file.write(key + " : " + str(value) + "\n")
    file.close()


def trail_all(models: List[Sequential], global_path_to_results, X_train, X_val,
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
        eval_f1(model, X_test, Y_test, path)
        print(function_name)

    stop = datetime.datetime.now()
    delta = stop - start
    print(delta)
