# -*- coding: utf-8 -*-

"""
  @author Nicolás Cubero

  Implementación del algoritmo de Moravec para la detección de esquinas

  Referencias:
  [1] -  Nilanjan Dey , Pradipti N , Nilanjana Barman , Debolina Das ,
        Subhabrata Chakraborty. A Comparative Study between Moravec and Harris
        Corner Detection of Noisy Images Using Adaptive Wavelet Thresholding
        Technique.  arXiv:1209.1558
"""

# Importación de librerías
import cv2 as cv
import numpy as np

# Para calcular variación de intensidad horizontal
__H_KERNEL = np.array([0, -1, 1])
__H_INV_KERNEL = np.array([1, -1, 0])

# Para calcular variación de intensidad vertical
__V_KERNEL = np.array([[1],
                        [-1],
                        [0]])

__V_INV_KERNEL = np.array([[0],
                        [-1],
                        [1]])

# Para calcular variación de intensidad en la diagonal 1
__DIAG1_KERNEL = np.array([[0, 0, 1],
                            [0, -1, 0],
                            [0, 0, 0]])

__DIAG1_INV_KERNEL = np.array([[0, 0, 0],
                            [0, -1, 0],
                            [1, 0, 0]])

# Para calcular variación de intensidad en la diagonal 2
__DIAG2_KERNEL = np.array([[0, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]])

__DIAG2_INV_KERNEL = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]])

__SUM_KERNEL = np.zeros((3,3)) + 1

def moravec(img: np.array, wind_size: int=3, thres: float=0.98):

    # Comprobar parámetros #
    #if not isinstance(img, np.ndarray):
    #    raise ValueError('La imagen de entrada no es una matriz válida')


    if not isinstance(wind_size, (int, np.uint8, np.int32, np.int64)) or wind_size <= 0:
        raise ValueError('"wind_size" debe de ser entero mayor que 0')

    if not isinstance(thres, float) or thres < 0 or thres > 1:
        raise ValueError('"thres" debe de ser un valor real entre 0 y 1')

    # Procedimiento #

    # Pasar la imagen a escala de grises
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('float32')

    # Calcular variaciones de intensidad horizontal, vertical, diagonales 1 y 2
    h = cv.filter2D(img, -1, __H_KERNEL)**2
    v = cv.filter2D(img, -1, __V_KERNEL)**2
    diag1 = cv.filter2D(img, -1, __DIAG1_KERNEL)**2
    diag2 = cv.filter2D(img, -1, __DIAG2_KERNEL)**2


    h_inv = cv.filter2D(img, -1, __H_INV_KERNEL)**2
    v_inv = cv.filter2D(img, -1, __V_INV_KERNEL)**2
    diag1_inv = cv.filter2D(img, -1, __DIAG1_INV_KERNEL)**2
    diag2_inv = cv.filter2D(img, -1, __DIAG2_INV_KERNEL)**2

    # Calcular la suma de variaciones en cada región
    sum_kernel = np.zeros((wind_size, wind_size)) + 1

    hh = cv.filter2D(h, -1, sum_kernel)
    vv = cv.filter2D(v, -1, sum_kernel)
    d1 = cv.filter2D(diag1, -1, sum_kernel)
    d2 = cv.filter2D(diag2, -1, sum_kernel)

    hh_inv = cv.filter2D(h_inv, -1, sum_kernel)
    vv_inv = cv.filter2D(v_inv, -1, sum_kernel)
    d1_inv = cv.filter2D(diag1_inv, -1, sum_kernel)
    d2_inv = cv.filter2D(diag2_inv, -1, sum_kernel)

    # Tomar el valor mínimo de cada variación
    c = np.minimum(
            np.minimum(hh, np.minimum(vv, np.minimum(d1, d2))),
            np.minimum(hh_inv, np.minimum(vv_inv, np.minimum(d1_inv, d2_inv)))
        )

    c = (c - c.min()) / (c.max()-c.min())

    # Suprimir los valores inferiores al umbral
    c[c < thres*c.max()] = 0

    # Método de no-máxima supresión
    # Calcular el máximo de cada región
    max_regions = np.zeros(img.shape)

    for i in range(max_regions.shape[0]):
        for j in range(max_regions.shape[1]):
            max_regions[i, j] = c[max(i-wind_size//2, 0):min(i+wind_size//2,
                                                                    c.shape[0]),
                                    max(j-wind_size//2, 0):min(j+wind_size//2,
                                                            c.shape[1])].max()

    # Aplicar no-máxima supresión
    maximals = np.isclose(c, max_regions) & (c > 0)
    maximals = np.argwhere(maximals)

    return maximals
