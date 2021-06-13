#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:49:24 2020

@author: nico
"""

# Importación de librerías
import numpy as np
import sys
from cv2.ml import SVM_load

class PedestrianDetector:
	
	"""Detecta peatones en una imagen mediante un clasificador SVM preentrenado
		basado en algún método de extracción de características
	"""
	
	def __init__(self, feat_extractor, model_filename):
		
		# Función que define el método de extracción de características
		self.__feat_extractor = feat_extractor
		
		# Cargar el modelo SVM
		self.__model = SVM_load(model_filename)
	
	
	def __make_pyramid(img: np.array, scale, minSize):
		
		# Definir el tamaño de las sucesivas imágenes
		sizes = []
		
		orig_h, orig_w = img.shape[:2] # Tomar el tamaño original
		
		# Calcular todos los tamaños
		new_h, new_w = orig_h, orig_w
		
		while new_h > minSize[0] and new_w > minSize[1]:
			# Calcular nuevas escalas
			new_h /= scale
			new_w /= scale
			
			sizes.append((new_h, new_w))
	
		
	
	def detect(img: np.array, scale=1.5, win_height=16, win_width=16,
			minSize=(128,64)):
		
		"""Recibe una imagen de entrada y detecta peatones devoliendo las
			coordenadas de los rectángulos que los engloban
		"""
		
		# Convetir la imagen en escala de grises
		gray = cvtColor(img, COLOR_BGR2GRAY)