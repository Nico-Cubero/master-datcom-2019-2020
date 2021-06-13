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
from steg_images import hideImage

# Leer argumentos pasados
parser = argparse.ArgumentParser(description='Revela una imagen '\
                                        'oculta en otra que tiene un determinado'\
                                        ' número de niveles d eprofundidad')
parser.add_argument('-i', '--image', help='Ruta de las imágen hospedadora de la'\
                                ' imagen oculta', type=str)
parser.add_argument('-o', '--ocult_img', help='Ruta de las imágen a ocultar'\
                                ' imagen oculta', type=str)
args = parser.parse_args()

img_filename = args.image
hid_filename = args.ocult_img


# Cargar la imagen
img = cv.imread(img_filename, cv.IMREAD_GRAYSCALE)

if img is None:
    print('No se pudo leer "{}"'.format(img_filename))
    exit(-1)

hid_img = cv.imread(hid_filename, cv.IMREAD_GRAYSCALE)

if hid_img is None:
    print('No se pudo leer "{}"'.format(hid_filename))
    exit(-1)

try:
    comp_img = hideImage(img, hid_img)
except Exception as e:
    print('Se produjo un error: ', str(e))
    exit(-1)

# Mostrar la imagen resultante
plt.imshow(comp_img, cmap='gray')
plt.show()
