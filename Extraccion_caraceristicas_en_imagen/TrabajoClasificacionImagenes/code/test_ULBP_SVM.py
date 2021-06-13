#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 05:12:50 2020

@author: nico

Script: test_LBP_SVM.py
Descripción: Clasifica las imágenes por medio de un clasificador SVM
	preentrenado y haciendo uso de descriptores LBP y las muestra por la
	pantalla junto a su clasificación.
Uso: test_HOG_SVM.py -m <Fichero dat con el modelo SVM>
						-i <Rutas de imágenes a clasificar,...>
"""

# Librerías incluídas
import sys
import argparse
import cv2 as cv
from cv2.ml import SVM_load
import matplotlib.pyplot as plt
from eci import compute_uniform_LBP
import numpy as np

#### Cuerpo del script ####

## Leer argumentos pasados al script
parser = argparse.ArgumentParser(description='Clasifica una imagen '\
								 'especificada por medio de un clasificador SVM'\
							 ' preentrenado y haciendo uso de descriptores LBP'\
							' uniformes')
parser.add_argument('-m', '--model', help='Fichero dat con el modelo '\
					'preentrenado', type=str)
parser.add_argument('-i', '--images', help='Conjunto de imágenes a clasificar',
					type=str, nargs='+')

args = parser.parse_args()

model_filename = args.model
images_filename = args.images


## Cargar el modelo SVM
try:
	svm = SVM_load(model_filename)
except Exception as e:
	print('Error al cargar el modelo:', file=sys.stderr)
	print(str(e), file=sys.stderr)
	exit(-1)

## Preparar datos auxiliares
labels = {1.0: 'pedestrian', 0.0: 'background'}


## Cargar y clasificar cada una de las imágenes
imgs = []
imgs_labels = []

for img_filename in images_filename:
	
	# Cargar la imagen candidata si es posible
	img = cv.imread(img_filename)
	
	if img is None:
		print('Error al cargar la imagen "{}"'.format(img_filename),
					file=sys.stderr)
		continue
	
	imgs.append(img)
	
	# Extraer descriptores LBP
	desc = compute_uniform_LBP(img)
	
	# Clasificar la imagen candidata
	img_label = svm.predict(desc.reshape(1,-1))[1][0][0]
	imgs_labels.append(labels[img_label])
	
## Mostrar las imágenes y su clasificación
fig = plt.figure()

for i in range(len(imgs)):
	
	a = fig.add_subplot(1, len(imgs), i+1)
	a.set_title(imgs_labels[i])
	plt.imshow(imgs[i])

	plt.gca().axes.get_xaxis().set_visible(False)
	plt.gca().axes.get_yaxis().set_visible(False)

plt.show()
