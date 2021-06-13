# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 02:39:28 2020

@author: nico
"""
import sys
from tensorflow.keras.callbacks import Callback
import numpy as np
from utils import mean_f1_score

class MeanF1Score_Evaluator(Callback):
	
	"""
		Callback para evaluar el F1 score medio
	"""
	
	def __init__(self, data: dict, show_conf_matrix=False):
		super(MeanF1Score_Evaluator, self).__init__()
		
		self.__data = data

		# Mostrar la matriz de confusión o no
		self.__show_conf_matrix = show_conf_matrix
	
	def on_train_begin(self, logs=None):

		self.__n_epochs = self.params['epochs']
		
		# Almacenar los resultados de la métrica evaluada sobre cada conjunto
		# de datos
		self.__results = dict()
		
		for d in self.__data:
			self.__results[d] = np.zeros((self.__n_epochs,))
			
	def on_epoch_end(self, epoch, logs=None):
		
		# Evaluar mean f1 score para cada conjunto de datos y en cada epoch
		for d in self.__data:
			y_pred = np.argmax(self.model.predict(self.__data[d]),
								  axis=1)
			#y_pred = self.model.predict(self.__data[d])
			y_true = self.__data[d].classes
			
			if self.__show_conf_matrix:
				score, conf_matrix = mean_f1_score(y_true, y_pred, True)

				print('Matriz de confusión: filas->real, columnas->predición')
				print('-'*20)
				np.savetxt(sys.stdout, conf_matrix, fmt='%2d')
			else:
				score = mean_f1_score(y_true, y_pred)
			
			self.__results[d][epoch] = score
			
			print('"{}" mean f1-score={}'.format(d, score))
	
	@property
	def results(self):
		return self.__results
