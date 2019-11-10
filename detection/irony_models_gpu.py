import inspect

from keras import Sequential
from keras.layers import Dense, Dropout, Bidirectional, CuDNNLSTM, Flatten
from sklearn import svm
from sklearn.metrics import accuracy_score


def baseline_00(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    clf = svm.SVC(gamma='scale')
    # clf = svm.SVC(gamma='scale', kernel='poly', degree=2)
    dataset_size = len(X_train)
    TwoDim_dataset = X_train.reshape(dataset_size, -1)
    clf.fit(TwoDim_dataset, Y_train)
    # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    #     decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    #     max_iter=-1, probability=False, random_state=None, shrinking=True,
    #     tol=0.001, verbose=False)

    X_val_reshape = X_val.reshape(len(X_val), -1)
    y_pred_val = clf.predict(X_val_reshape)
    print("accuracy_score_val")
    print(accuracy_score(Y_val, y_pred_val))

    X_test_reshape = X_test.reshape(len(X_test), -1)
    y_pred_test = clf.predict(X_test_reshape)
    print("accuracy_score_test")
    print(accuracy_score(Y_test, y_pred_test))


def give_model_00(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(Dense(20, input_shape=(max_sentence_length, len_of_vector_embeddings)))
    # model.add(Dense(20))
    # model.add(Dense(20))
    # model.add(Dense(40))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_10(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(CuDNNLSTM(10, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_20(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(CuDNNLSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_30(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(CuDNNLSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(CuDNNLSTM(20, return_sequences=False))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_40(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(CuDNNLSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_50(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(CuDNNLSTM(10, return_sequences=True),
                      input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(10, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_60(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(CuDNNLSTM(20, return_sequences=True),
                      input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(20, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


##################################################
# CuDNNLSTM vs bi_lstm
def give_model_41(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(CuDNNLSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(20, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_61(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(CuDNNLSTM(20, return_sequences=True),
                      input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(20, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(20, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_50000(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(CuDNNLSTM(20, return_sequences=True),
                      input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(20, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name


def give_model_50001(len_of_vector_embeddings, max_sentence_length):
    model: Sequential = Sequential()
    model.add(
        Bidirectional(CuDNNLSTM(100, return_sequences=True),
                      input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(CuDNNLSTM(100, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    function_name = inspect.currentframe().f_code.co_name
    return model, function_name
