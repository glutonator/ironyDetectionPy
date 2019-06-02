from detection.data_inputs import give_data
from detection.irony_models import model_ooo, give_model

max_sentence_length = 15
X_train, X_test, Y_train, Y_test = give_data(max_sentence_length)
len_of_vector_embeddings = 25

model = give_model(len_of_vector_embeddings, max_sentence_length)
model_ooo(X_train, X_test, Y_train, Y_test, model)
