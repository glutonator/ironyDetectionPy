import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from detection.my_plots import generate_plots


def give_model(len_of_vector_embeddings, max_sentence_length):
    model = Sequential()
    # model.add(LSTM(32, input_shape=(2, 10), return_sequences=True))
    # model.add(LSTM(20, input_shape=(11, 25), return_sequences=False))
    # model.add(LSTM(20, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=False))
    # model.add(LSTM(50, input_shape=(max_sentence_length, len_of_vector_embeddings), return_sequences=True))
    model.add(
        Bidirectional(LSTM(20, return_sequences=True), input_shape=(max_sentence_length, len_of_vector_embeddings)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(20, return_sequences=False)))
    # model.add(LSTM(20, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def model_ooo(X_train, X_test, Y_train, Y_test, model):

    save_best = keras.callbacks.ModelCheckpoint('weights.best.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    results = model.fit(X_train, Y_train, validation_split=0.2,
                        callbacks=[early_stop, save_best], epochs=30, batch_size=10,
                        verbose=0)

    generate_plots(results)

    scores = model.evaluate(X_test, Y_test, verbose=1)
    print(model.metrics_names)
    print(scores)
