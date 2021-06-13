# -*- coding: utf-8 -*-

"""
  @author Nicolás Cubero

  Implementación de los métodos para la ocultación y descubrimiento
  de imágenes en imágenes
"""

# Librerías importadas
import cv2 as cv
import numpy as np

def hideImage(hosp_img: np.array, hid_img: np.array):

    """
        Oculta la imagen hid_img en la imagen hosp_img
    """

    # Comprobar parámetros #
    if len(hosp_img.shape) != 2 or hosp_img.dtype != np.uint8:
        raise ValueError('La imagen hospedadora no es una imagen en escala de grises válida')

    if len(hid_img.shape) != 2 or hid_img.dtype != np.uint8:
        raise ValueError('La imagen a ocultar no es una imagen en escala de grises válida')

    if hosp_img.shape != hid_img.shape:
        raise ValueError('Las dimensiones de la imagen hospedadora y la imagen a ocultar deben de ser iguales')

    # Procedimiento #

    # Determinar los niveles necesarios para ocultar la imagen
    n_levels = np.unique(hid_img).size

    if n_levels < 2:
        return np.zeros((hosp_img.shape[0], hosp_img.shape[1]), dtype='uint8')

    # Reescalar los valores de la intensidad de la imagen al conjunto mínimo
    # de valores necesarios para representar los diferentes niveles
    factor = hid_img.max()/n_levels
    hid_img = np.around(hid_img / factor).astype('uint8')


    mask = (1 << (n_levels-1)) - 1
    # Suprimir los "mask" últimos bits de la imagen hospedadora para almacenar
    # en ellos los valores de la imagen a ocultar

    ret_img = ((hosp_img & (~mask)) | hid_img).astype('uint8')

    return ret_img

def revealImage(hosp_img: np.array, n_levels: int=2):

    """
        Revela la imagen oculta en la imagen hosp_img considerando n_levels
        niveles de gris para la imagen oculta
    """

    # Comprobar parámetros #
    if n_levels < 2:
        raise ValueError('El número de niveles de colores debe de ser al menos de 2')

    if len(hosp_img.shape) != 2 or hosp_img.dtype != np.uint8:
        raise ValueError('La imagen hospedadora no es una imagen en escala de grises válida')

    # Procedimiento #

    # Determinar el número de bits que ocultan la intensidad de la imagen oculta
    remain = 1 << (n_levels-1)

    # Extraer la información de la imagen oculta
    hid_img = hosp_img % remain

    # Extender los niveles de intensidad de la imagen oculta
    factor = 255/hid_img.max()
    hid_img = (hid_img * factor).astype('uint8')

    return hid_img
