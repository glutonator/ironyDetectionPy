from typing import List
from tensorflow_core.python.keras import Sequential

import detection.irony_models_gpu as di_gpu
import detection.irony_models_cpu as di_cpu


def get_all_models_gpu(total_length, max_sentence_length):
    models: List[Sequential] = []

    # models.append(di_gpu.give_model_8000_cnn(total_length, max_sentence_length))
    # models.append(di_gpu.give_model_8001_cnn(total_length, max_sentence_length))
    # models.append(di_gpu.give_model_8002_cnn(total_length, max_sentence_length))
    # models.append(di_gpu.give_model_8010_cnn(total_length, max_sentence_length))
    # models.append(di_gpu.give_model_8011_cnn(total_length, max_sentence_length))
    # models.append(di_gpu.give_model_8012_cnn(total_length, max_sentence_length))

    models.append(di_gpu.give_model_00(total_length, max_sentence_length))
    models.append(di_gpu.give_model_10(total_length, max_sentence_length))
    models.append(di_gpu.give_model_20(total_length, max_sentence_length))
    models.append(di_gpu.give_model_30(total_length, max_sentence_length))
    models.append(di_gpu.give_model_40(total_length, max_sentence_length))
    models.append(di_gpu.give_model_50(total_length, max_sentence_length))
    models.append(di_gpu.give_model_60(total_length, max_sentence_length))

    models.append(di_gpu.give_model_41(total_length, max_sentence_length))
    models.append(di_gpu.give_model_61(total_length, max_sentence_length))

    models.append(di_gpu.give_model_50000(total_length, max_sentence_length))
    models.append(di_gpu.give_model_50001(total_length, max_sentence_length))

    return models


def get_all_models_cpu(total_length, max_sentence_length):
    models: List[Sequential] = []

    # models.append(di_cpu.give_model_10(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_20(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_30(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_40(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_50(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_60(total_length, max_sentence_length))
    #
    # models.append(di_cpu.give_model_41(total_length, max_sentence_length))
    # models.append(di_cpu.give_model_61(total_length, max_sentence_length))
    #
    # models.append(di_cpu.give_model_50000(total_length, max_sentence_length))
    models.append(di_cpu.give_model_50001(total_length, max_sentence_length))

    return models
