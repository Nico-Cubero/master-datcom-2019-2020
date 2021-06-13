# -*- coding: utf-8 -*-

"""
  @author Nicolás Cubero

  Visión por Computador
  Trabajo 3 - Algoritmo de Moravec
  Fecha: 7 de Junio de 2020
"""

# Importar librerías
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moravec import moravec

# Constantes
MAX_IMAGES_PER_FIG = 6

MAX_IMAGES_PER_ROW = 2
MAX_IMAGES_PER_COL = 3

# Leer argumentos pasados
parser = argparse.ArgumentParser(description='Detecta bordes en imágenes '\
                                        'el algoritmo de Moravec y los muestra'\
                                        ' junto a las imágenes')
parser.add_argument('-w', '--wsize', help='Tamaño de ventana a usar',
                    type=int, default=3)
parser.add_argument('-t', '--thres', help='Ratio de umbral permitido',
                    type=float, default=0.2)
parser.add_argument('-i', '--images', help='Ruta de las imágenes a clasificar',
					type=str, nargs='+')

args = parser.parse_args()

wsize = args.wsize
thres = args.thres
images_filename = args.images

# Cargar imágenes
imgs = []

for img_filename in images_filename:
    img = cv.imread(img_filename)

    if img is None:
        print('No se pudo leer "{}"'.format(img_filename))
    else:
        imgs.append(img)

# Detectar bordes con Moravec en todas las imágenes
corners = [moravec(img, wsize, thres) for img in imgs]

# Representarlos en pantalla
for i in range(0, len(corners), MAX_IMAGES_PER_FIG):

    fig = plt.figure()

    # Determinar las imágenes que se mostrarán por fila y columna
    rang = range(i,min(MAX_IMAGES_PER_FIG+i, len(imgs)))

    n_cols = (len(rang) % MAX_IMAGES_PER_COL) if len(rang)//MAX_IMAGES_PER_COL < 1 else MAX_IMAGES_PER_COL
    n_rows = (len(rang) // MAX_IMAGES_PER_ROW) + int((len(rang) % MAX_IMAGES_PER_ROW) > 0)


    for j in rang:

        plt.subplot(n_rows*100 + n_cols*10 + j + 1)

        # Representar la imagen con los corners
        plt.imshow(imgs[j])
        plt.plot(corners[j][:,1], corners[j][:,0], 'r.')

    plt.show()
