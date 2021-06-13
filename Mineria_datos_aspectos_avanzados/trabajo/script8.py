# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:28:01 2020

@author: nico

Permite entrenar un modelo de Deep Learning que identifica 54 clases de imágenes
del suelo capturadas desde satélite. Las imágenes son cargadas en formato RGB
y cada uno de sus píxeles es reescalado a [0,1].
Este script admite parametrización en el número de capas convolucionales,
el número de filtros usados por las mismas, tamaño de kernel, stride y uso de
max pooling 2d de tamaño 2 tras cada capa. Para las capas densas, permite
especificar el número de componentes y el Dropout de cada capa.

Los entrenamientos se ejecutan sobre la GPU.
El script hace uso de Data Augmentation para duplicar el conjunto de datos de
train
Se añade la callback ReduceLROnplateau que disminuye el learning rate del
optimizador Adam en un factor 0.5 si el val_loss no disminuye en 5 épocas.


Uso: script8.py -d <Documento JSON con los experimentos a ejecutar> [-s]

El documento JSON con los experimentos deberá de presentar el siguiente formato:
- epochs -> int: Número de épocas del entrenamiento
- batch_size -> int o list de int: Tamaños de batch a considerar
- conv2d: Con los siguientes parámetros:
	· num_filters -> list de int: Lista de enteros con los números de filtros de
					cada una de las capas convolucionales.
	· kernel_size -> list de int: Lista de enteros con los tamaños de kernel de
					cada una de las capas convolucionales.
	· stride -> list de int: Lista de enteros con el stride de cada capa
				convolucional.
	· max_pooling -> list de 0, 1: Para cada capa especificar 1 si se desea que
					esté sucedida por una capa de Max Pooling 2D de tamaño 2 y
					con stride 2.

	Nota: El tamaño de las listas de los anteriores parámetros debe de coincidir

- dense: Con los siguientes parámetros:
	· layers -> list de int: Con el número de entradas de cada capa densa.
	· dropout -> list de float: Con el Dropout de cada capa densa

	Nota: El tamaño de las listas de los anteriores parámetros debe de coincidir

Nota: Para aquellos parámetros en los que se pase una lista de valores, se
	realizará una experimentación para cada valor del parámetro combinándose
	con cada uno de los valores del resto de parámetros.

Ejemplo:

	{
		"epochs": 30
		"batch_size": [100, 200],
		"conv2d": [		{
					"num_filters": [14, 30, 44, 64],
					"kernel_size": [5, 3, 3, 2],
					"stride": [2, 2, 1, 1],
					"max_pooling": [1, 1, 1, 1]
					},
	{
					"num_filters": [20, 32, 44, 64, 96, 128],
					"kernel_size": [3, 3, 3, 3, 2, 1],
					"stride": [2, 2, 1, 1, 1, 1],
					"max_pooling": [0, 0, 1, 1, 1, 1]
					}
				],
		"dense": [
					{
						"layers": [70, 64],
						"dropout": [0.12, 0.11]
					}
				]
	}
"""

TRAIN_IMGS_DIR = './train/train/'
VAL_IMGS_DIR = './validacion/validacion/'

import sys
import json
import utils
import argparse
import models
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from meanF1Score_Evaluator import MeanF1Score_Evaluator
from keras.callbacks.callbacks import (History, ModelCheckpoint,
														   ReduceLROnPlateau)
#import keras.backend as K

np.random.seed(11) # Semilla generadora de números aleatorios

## Leer argumentos pasados al script
parser = argparse.ArgumentParser(description='Entrena modelo de Deep Learning '\
								 'que identifica imágenes del suelo')
parser.add_argument('-d', '--document', help='Fichero JSON con los parámetros'\
					' del entrenamiento a considerar', type=str)
parser.add_argument('-s', '--save_model', help='Establecer para almacenar los'\
					' modelos generados durante el entrenamiento',
					action='store_true', default=False)

args = parser.parse_args()

exp_filename = args.document
store_models = args.save_model

# Leer fichero de experimentación
with open(exp_filename) as f:
	try:
		exp_data = json.load(f)
	except Exception as e:
		print('Se produjo un error con el fichero de experimentación'\
			' expecificado:\n',str(e), file=sys.stderr)
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

## Extracción de todas las combinaciones de parámetros
params = utils.extract_experiments_parameters(exp_data)
results = []

for p in params:
	print('Entrenando con parámetros: {}'.format(p))
	np.random.seed(11) # Semilla generadora de números aleatorios

	## Carga de imágenes de entrenamiento
	loader = ImageDataGenerator(rescale=1./255)
	
	train_data = loader.flow_from_directory(TRAIN_IMGS_DIR,
											class_mode='sparse',
											batch_size=p['batch_size'],
											color_mode='rgb')
	
	val_data = loader.flow_from_directory(VAL_IMGS_DIR, class_mode='sparse',
											 batch_size=p['batch_size'],
											 color_mode='rgb',
											 shuffle=False)

	## Preparar imágenes aumentadas
	aug_loader = ImageDataGenerator(rescale=1./255, rotation_range=270,
								 zoom_range=(0.5, 1.5),
								 shear_range=0.3,
								 brightness_range=(0.4, 1.5),
								 horizontal_flip=True)

	train_aug_data = aug_loader.flow_from_directory(TRAIN_IMGS_DIR,
											class_mode='sparse',
											batch_size=p['batch_size'],
											color_mode='rgb')

	combiner = utils.GenetarorCombiner((train_data, train_aug_data))
	train_full = combiner.generate()

	## Cargar arquitectura y compilar modelo
	model = models.model67_sep_conv_dense(conv_conf=p['conv2d'],
											dense_conf=p['dense'])
	
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	## Preparar evaluador de f1 score medio
	mean_f1_score_eval = MeanF1Score_Evaluator({'val': val_data},
												show_conf_matrix=True)
	## Preparar el registro de la evolución de las métricas
	history = History()

	## Preparar el registro del mejor modelo
	checkpoint = ModelCheckpoint(filepath=model_base_filename + '-experimento-' +
								str(len(results)+1) +
								'_modelo_backup.h5',
								monitor='val_loss',
								save_best_only=True)

	## Ajuste dinámico del learning rate
	rlrp = ReduceLROnPlateau(factor=0.1, patience=5, verbose=True,
						  min_delta=0.005, min_lr=1e-8)

	## Entrenamiento de los modelos
	try:
		model.fit(train_full, verbose=True,
				epochs=p['epochs'],
				steps_per_epoch=len(combiner),
				validation_data=val_data,
				callbacks=[history, mean_f1_score_eval, checkpoint, rlrp])
	except Exception as e:
		p['resultados'] = 'Error al realizar el experimento: '+str(e)
		results.append(p)

		# Almacenar los datos en un JSON
		with open(results_filename, 'w') as f:
			json.dump(results, f, indent=4)

		continue

	# Obtener la evaluación del f1 score
	mean_f1_score = mean_f1_score_eval.results
	
	## Evaluar el modelo
	score_train = model.evaluate(train_data)
	score_test = model.evaluate(val_data)

	# Almacenar gráfica de accuracy y loss
	utils.plot_results({'accuracy - train': history.history['accuracy'],
				   'accuracy - validation': history.history['val_accuracy']},
					'Accuracy',
					model_base_filename + '-experimento-'+str(len(results)+1) +
																'_accuracy.pdf')

	utils.plot_results({'loss - train': history.history['loss'],
				   'loss - validation': history.history['val_loss']},
					'Loss',
					model_base_filename + '-experimento-'+str(len(results)+1) +
																'_loss.pdf')

	utils.plot_results({'Mean F1-Score - validation': mean_f1_score['val']},
					'Mean F1-Score',
					model_base_filename + '-experimento-'+str(len(results)+1) +
														'_mean_f1_score.pdf')

	# Almacenar el modelo
	if store_models:
		model.save(model_base_filename + '-experimento-'+str(len(results)+1) +
																'_modelo.h5')
	
	# Almacenar predición
	y_pred = np.argmax(model.predict(val_data), axis=1)
	val_filename = [name.split('/')[-1]  for name in val_data.filepaths]

	utils.save_prediction(ids=val_filename, labels=y_pred,
						   filename=model_base_filename + '-experimento-' +
										str(len(results)+1) +'_prediction.csv')

	# Almacenar toda la información de parámetros y resultados
	p['script'] = __file__
	p['arquitectura'] = model.get_config()
	p['resultados'] = {'loss_train': float(score_train[0]),
						  'accuracy_train': float(score_train[1]),
						'loss_test': float(score_test[0]),
						  'accuracy_test': float(score_test[1]),
						  'ratio_loss_train_val': float(score_train[0])/
													float(score_test[0]),
						  'mean_f1_score_test': float(mean_f1_score['val'][-1])}
	results.append(p)

	# Almacenar los datos en un JSON
	with open(results_filename, 'w') as f:
		json.dump(results, f, indent=4)
