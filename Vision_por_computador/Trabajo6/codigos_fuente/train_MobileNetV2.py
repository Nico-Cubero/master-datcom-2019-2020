# -*- coding: utf-8 -*-
"""

@author: nico

Script: train_MobileNetV2.py
Descripción: Entrena una red CNN partiendo del modelo MobileNetV2 preentrenado
            para clasificar imágenes en las que se muestre un peatón de aquellas
            en las que sólo aparezca fondo.
Uso: train_MobileNetV2.py -d <Documento JSON de experimentación> [-s]

El documento JSON con los experimentos deberá de presentar el siguiente formato:
- epochs -> int: Número de épocas del entrenamiento
- batch_size -> int o list de int: Tamaños de batch a considerar
- lr -> float o lista de floats con los learning rates a considerar
- arquitectura: Conjunto de capas que se añadirán al final del modelo
preentrenado. Se debe de especificar como un modelo keras en formato JSON

Nota: Para aquellos parámetros en los que se pase una lista de valores, se
	realizará una experimentación para cada valor del parámetro combinándose
	con cada uno de los valores del resto de parámetros.

Por su parte, si se especifica el parámetro -s, todo modelo generado se
almacenará.

"""

# Librerías incluídas
import sys
import time
import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, confusion_matrix)
import tensorflow as tf
from keras.callbacks.callbacks import History
from utils import (load_image_class_dataset, compute_HOG, train_svm, test_svm,
                    check_experiment_document, extract_experiments_cnn_parameters,
                    no_pederestian_img_prepro, model_transfer_MobileNetV2, plot_results)

PATH_POSITIVE = '../data/pedestrians128x64/'
PATH_NEGATIVE = '../data/pedestrians_neg/'

#### Cuerpo del script ####

## Leer argumentos pasados al script
parser = argparse.ArgumentParser(description='Entrena un red CNN partiendo de'\
                                 ' la red MobileNetV2 para identificar imágenes con '\
                                 ' peatones e imágenes de fondo')
parser.add_argument('-d', '--document', help='Fichero JSON con los parámetros'\
                    ' del entrenamiento a considerar', type=str)
parser.add_argument('-s', '--save_model', help='Establecer para almacenar los'\
                    ' modelos generados durante el entrenamiento',
                    action='store_true', default=False)

args = parser.parse_args()

exp_filename = args.document
store_models = args.save_model

np.random.seed(27) # Semilla inicializadora

# Leer fichero de experimentación
with open(exp_filename) as f:
    try:
        exp_data = json.load(f)
    except Exception as e:
        print('Se produjo un error con el fichero de experimentación'\
            ' expecificado:\n',str(e), file=sys.stderr)
        exit(-1)

# Comprobar el formato del fichero de documentación
try:
    check_experiment_document(exp_data)
except Exception as e:
    print('El documento de experimentación no es correcto:', file=sys.stderr)
    print(str(e), file=sys.stderr)
    exit(-1)

# Determinar el nombre del fichero destino y del modelo destino
dot_pos = exp_filename.rfind('.')
if dot_pos != -1:
    results_filename = exp_filename[:dot_pos] + '_experimentos.json'
    model_base_filename = exp_filename[:dot_pos]
else:
    results_filename = exp_filename + '_experimentos.json'
    model_base_filename = exp_filename[:]

## Deshabilitar GPU (por ahora)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

## Ajustar el uso máximo de la GPU y la CPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

## Carga de imágenes

# Cargar las imágenes
print('Cargando imágenes: ')
X, y = load_image_class_dataset(data={1: PATH_POSITIVE, 0: PATH_NEGATIVE},
                                prepro_operation={0: no_pederestian_img_prepro},
                                n_samples={1: 400},
                                rep_samples={0: 400//50})
print('Cargadas con éxito ',X.shape[0],' imágenes')

# Dividir en conjunto de train y test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Extracción de todas las combinaciones de parámetros
params = extract_experiments_cnn_parameters(exp_data)
results = []

for p in params:
    print('Entrenando con parámetros: {}'.format(p))

    # Cargar la arquitectura y compilar
    model = model_transfer_MobileNetV2(p['arquitectura'], p['load_weights'])[0]

    # Preparar el registro de la evolución de las métricas
    history = History()

    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
    optimizer='adam')


    # Entrenar modelos
    try:
        model.fit(X_train, y_train, verbose=True,
                epochs=p['epochs'],
                callbacks=[history])
    except Exception as e:
        p['resultados'] = 'Error al realizar el experimento: '+str(e)
        results.append(p)

        # Almacenar los datos en un JSON
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)

            continue

    ## Evaluar el modelo
    train_pred = model.predict(X_train).astype('int64')
    test_pred = model.predict(X_test).astype('int64')
    score_train = model.evaluate(X_train, y_train)
    score_test = model.evaluate(X_test, y_test)

    # Almacenar gráfica de accuracy y loss
    plot_results({'accuracy - train': history.history['accuracy']},
                    'Accuracy',
                    model_base_filename + '-experimento-'+str(len(results)+1) +
                                                                '_accuracy.pdf')

    plot_results({'loss - train': history.history['loss']},
                    'Loss',
                    model_base_filename + '-experimento-'+str(len(results)+1) +
                                                                '_loss.pdf')

    # Almacenar el modelo
    if store_models:
        model.save(model_base_filename + '-experimento-'+str(len(results)+1) +
                                                                '_modelo.h5')

    conf_matrix_train = confusion_matrix(y_train, train_pred)
    conf_matrix_test = confusion_matrix(y_test, test_pred)

    # Almacenar toda la información de parámetros y resultados
    p['script'] = __file__
    #p['arquitectura'] = model.get_config()
    p['resultados'] = {'loss_train': float(score_train[0]),
                          'accuracy_train': float(score_train[1]),
                        'loss_test': float(score_test[0]),
                          'accuracy_test': float(score_test[1]),
                          'ratio_loss_train_val': float(score_train[0])/
                                                    float(score_test[0]),
                        'precision_train': precision_score(y_train, train_pred),
                        'precision_test': precision_score(y_test, test_pred),
                        'recall_train': recall_score(y_train, train_pred),
                        'recall_test': recall_score(y_test, test_pred),
                        'f1_train': f1_score(y_train, train_pred),
                        'f1_test': f1_score(y_test, test_pred),
                        'confusion_matrix_train': {
                                                    'TP': int(conf_matrix_train[1,1]),
                                                    'TN': int(conf_matrix_train[0,0]),
                                                    'FP': int(conf_matrix_train[0,1]),
                                                    'FN': int(conf_matrix_train[1,0])
                                                },
                        'confusion_matrix_test': {
                                                    'TP': int(conf_matrix_test[1,1]),
                                                    'TN': int(conf_matrix_test[0,0]),
                                                    'FP': int(conf_matrix_test[0,1]),
                                                    'FN': int(conf_matrix_test[1,0])
                                                }
                    }

    results.append(p)

    # Almacenar los datos en un JSON
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
