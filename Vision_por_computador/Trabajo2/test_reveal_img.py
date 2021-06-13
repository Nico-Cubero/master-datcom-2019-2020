# -*- coding: utf-8 -*-

"""
  @author Nicolás Cubero

  Visión por Computador
  Trabajo 2 - Lo que oculta una imagen
  Fecha: 7 de Junio de 2020
"""

# Importar librerías
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from steg_images import revealImage

# Leer argumentos pasados
parser = argparse.ArgumentParser(description='Revela una imagen '\
                                        'oculta en otra que tiene un determinado'\
                                        ' número de niveles d eprofundidad')
parser.add_argument('-l', '--levels', help='Niveles de profundidad de la imagen'\
                    ' oculta', type=int, default=2)
parser.add_argument('-i', '--image', help='Ruta de las imágen hospedadora de la'\
                                ' imagen oculta', type=str)
args = parser.parse_args()

levels = args.levels
img_filename = args.image

if levels < 2:
    print('La imagen oculta no puede tener menos de 2 niveles de profundidad')
    exit(-1)

# Cargar la imagen
img = cv.imread(img_filename, cv.IMREAD_GRAYSCALE)

if img is None:
    print('No se pudo leer "{}"'.format(img_filename))
    exit(-1)

hid_img = revealImage(img, levels)

# Mostrar la imagen resultante
plt.imshow(hid_img, cmap='gray')
plt.show()
