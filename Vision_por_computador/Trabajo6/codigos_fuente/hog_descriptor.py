# -*- coding: utf-8 -*-

"""
  @author Nicolás Cubero

  Implementación del método HOG (Histogram of Gradient) para la extracción de
  características en imágenes
"""

# Librerías importadas
import numpy as np
import cv2 as cv
import sys
# Variables globales
__H_FILTER = np.array([[-1, 0, 1]])
__V_FILTER = np.array([[-1],
                        [0],
                        [1]])
__CELL_ROW = 8
__CELL_COL = 8
__INTERVALS = np.linspace(10, 180, 9)

__CELLS_PER_BLOCK_ROW = 2
__CELLS_PER_BLOCK_COL = 2

def __do_cartesian_product(a: np.array, b: np.array):
	return np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])

normL2 = lambda x, eps=1e-7: x/np.sqrt((x**2).sum() + eps)

def __cellHOG(mag: np.array, angle: np.array, i: int, j: int,
                cell_rows: int, cell_cols: int, intervals: np.array):

    # Preparar el histograma
    hist = np.zeros((intervals.size))

    # Tomar el área seleccionada
    mag = mag[i:i+cell_rows, j:j+cell_cols]
    angle = angle[i:i+cell_rows, j:j+cell_cols]

    for (ii, jj), v in np.ndenumerate(angle):

        eq_limit = (intervals == v)

        if eq_limit.any():
            # Acumular los elementos iguales a los límites
            hist[eq_limit] += mag[ii, jj]
        else:
            # Acumular los elementos entre límites
            for k in np.arange(intervals.size):
                if v > intervals[k] and v < intervals[(k+1)%intervals.size]:
                    hist[k] += mag[ii, jj]*(intervals[(k+1)%intervals.size] - v)/(intervals[(k+1)%intervals.size] - intervals[k])
                    hist[(k+1)%intervals.size] += mag[ii, jj]*(v - intervals[k])/(intervals[(k+1)%intervals.size] - intervals[k])

                    break

    return hist

# Vectorizar esta función
__vect_cellHOG = np.vectorize(__cellHOG, excluded=[0,1,4,5,6], signature='(),()->(n)')

def __block_norm(hist_cells: np.array, i: int, j: int,
                            cells_per_row, cells_per_col):

    # Tomar histogramas de las celdas que entran en el bloque
    block = hist_cells[i:i+cells_per_row, j:j+cells_per_col, :].reshape(-1)

    # Realizar normalización
    block = normL2(block)

    return block

# Vectorizar esta función
__vect_block_norm = np.vectorize(__block_norm, excluded=[0,3,4], signature='(),()->(n)')

def comp_HOGDescriptor(img: np.array):

    """
        Computa los descriptores HOG de una imagen
    """

    img = img.astype('float32')

    # Calcular gradientes verticales y horizontales
    gx = cv.filter2D(img, -1, __H_FILTER)
    gy = cv.filter2D(img, -1, __V_FILTER)


    # Calcular la magnitud y orientación de los gradientes
    mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)


    # Quedarnos con la variación de mayor magnitud de todas las bandas
    sum_mag = np.sqrt((mag**2).sum(axis=(0,1)))
    max_mag = np.argmax(sum_mag)

    mag = mag[:,:,max_mag]
    angle = angle[:,:,max_mag]

    # Añadir 90º para obtener el ángulo de cada arista.
    angle += 90
    angle %= 180


    # Calcular histograma asociado a cada celda
    index = __do_cartesian_product(np.arange(0, img.shape[1], __CELL_COL),
                                    np.arange(0, img.shape[0], __CELL_ROW)
                                ).astype('int64')

    cells = __vect_cellHOG(mag, angle, index[:,1], index[:,0],__CELL_ROW,
                            __CELL_COL, __INTERVALS).reshape(img.shape[0]//__CELL_ROW,
                                                            img.shape[1]//__CELL_COL, -1)


    # Calcular valor de cada bloque
    index = __do_cartesian_product(np.arange(cells.shape[0]-__CELLS_PER_BLOCK_ROW+1),
                                np.arange(cells.shape[1]-__CELLS_PER_BLOCK_COL+1)).astype('int64')

    blocks = __vect_block_norm(cells, index[:, 0], index[:, 1],
                                __CELLS_PER_BLOCK_ROW, __CELLS_PER_BLOCK_COL)

    # Calcular descriptor HOG
    hog_desc = blocks.reshape(-1)
    hog_desc = normL2(hog_desc) #/= np.sqrt((np.sum(hog_desc**2) + 1e-7))

    # Saturar los valores mayores que 0.2
    hog_desc[hog_desc>0.2] = 0.2

    return hog_desc.astype('float32')
