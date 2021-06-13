# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:28:01 2020

@author: nico


Permite entrenar un modelo de Deep Learning que identifica 54 clases de imágenes
del suelo capturadas desde satélite. Las imágenes son cargadas en formato RGB
y cada uno de sus píxeles es reescalado a [0,1].
Este script admite parametrización en el número de filtros de las 4 primeras
capas convolucionales, basándose en los modelos del modelo 66 de models.py

Los entrenamientos se ejecutan sobre la GPU.


Uso: script4.py -d <Documento JSON con los experimentos a ejecutar> [-s]

El documento JSON con los experimentos deberá de presentar el siguiente formato:
- epochs -> int: Número de épocas del entrenamiento
- batch_size -> int o list de int: Tamaños de batch a considerar
- conv2d_1_filters -> int o list de int: Número de filtros a considerar para la
										primera capa convolucional en las
										diferentes experimentaciones.
- conv2d_2_filters -> int o list de int: Número de filtros a considerar para la
										segunda capa convolucional en las
										diferentes experimentaciones. 
- conv2d_3_filters -> int o list de int: Número de filtros a considerar para la
										tercera capa convolucional en las
										diferentes experimentaciones. 
- conv2d_4_filters -> int o list de int: Número de filtros a considerar para la
										cuarta capa convolucional en las
										diferentes experimentaciones. 

Nota: Para aquellos parámetros en los que se pase una lista de valores, se
	realizará una experimentación para cada valor del parámetro combinándose
	con cada uno de los valores del resto de parámetros.

Ejemplo:

	{
		"epochs": 30
		"batch_size": [100, 200],
		"conv2d_1_filters": 20,
		"conv2d_2_filters": 32,
		"conv2d_3_filters": 48,
		"conv2d_4_filters": 54,
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
from keras.callbacks.callbacks import History, ModelCheckpoint

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
	
	## Cargar arquitectura y compilar modelo
	model = models.model66_conv2d_filters(p)
	
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

	## Preparar evaluador de f1 score medio
	mean_f1_score_eval = MeanF1Score_Evaluator({'val': val_data},
												show_conf_matrix=True)
	## Preparar el registro de la evolución de las métricas
	history = History()

	## Preparar el registro del mejor modelo
	checkpoint = ModelCheckpoint(filepath=model_base_filename +
									'-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_modelo_backup.h5',
								monitor='val_loss',
								save_best_only=True)

	## Entrenamiento de los modelos
	model.fit(train_data, verbose=True,
			epochs=p['epochs'],
			validation_data=val_data,
			callbacks=[history, mean_f1_score_eval, checkpoint])

	# Obtener la evaluación del f1 score
	mean_f1_score = mean_f1_score_eval.results
	
	## Evaluar el modelo
	score_train = model.evaluate(train_data)
	score_test = model.evaluate(val_data)

	# Almacenar gráfica de accuracy y loss
	utils.plot_results({'accuracy - train': history.history['accuracy'],
				   'accuracy - validation': history.history['val_accuracy']},
					'Accuracy',
					model_base_filename + '-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_accuracy.pdf')

	utils.plot_results({'loss - train': history.history['loss'],
				   'loss - validation': history.history['val_loss']},
					'Loss',
					model_base_filename + '-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_loss.pdf')

	utils.plot_results({'Mean F1-Score - validation': mean_f1_score['val']},
					'Mean F1-Score',
					model_base_filename + '-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_mean_f1_score.pdf')

	# Almacenar el modelo
	if store_models:
		model.save(model_base_filename + '-'.join((str(k) + '=' +
								  str(p[k]) for k in p)) + '_modelo.h5')
	
	# Almacenar predición
	y_pred = np.argmax(model.predict(val_data), axis=1)
	val_filename = [name.split('/')[-1]  for name in val_data.filepaths]

	utils.save_prediction(ids=val_filename, labels=y_pred,
						   filename=model_base_filename + '-'.join((str(k) +
							'=' + str(p[k]) for k in p)) + '_prediction.csv')

	# Almacenar toda la información de parámetros y resultados
	p['script'] = __file__
	p['arquitectura'] = model.get_config()
	p['resultados'] = {'loss_train': float(score_train[0]),
						  'accuracy_train': float(score_train[1]),
						'loss_test': float(score_test[0]),
						  'accuracy_test': float(score_test[1]),
						  'mean_f1_score_test': float(mean_f1_score['val'][-1])}
	results.append(p)

	# Almacenar los datos en un JSON
	with open(results_filename, 'w') as f:
		json.dump(results, f, indent=4)
