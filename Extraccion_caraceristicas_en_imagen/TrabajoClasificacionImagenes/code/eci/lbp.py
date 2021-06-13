# -*- coding: utf-8 -*-

# Importación de librerías
import numpy as np
from cv2 import cvtColor, COLOR_BGR2GRAY

class LBPDescriptor:
	
	"""Clase LBPDescriptor
	
		Utilidad para el cómputo de descriptores basados en Local Binary Pattern
	"""

	# Tamaños de ventanas, celdas y de desplazamiento
	__WIND_WIDTH = 64
	__WIND_HEIGHT = 128
	
	__CELL_SIZE = 16
	
	__V_SHIFT = 8
	__H_SHIFT = 8


	# Para el cálculo del valor decimal asociado a cada vecindad
	__MASK_CELL = np.array([[1<<7, 1<<6, 1<<5],
							  [1, 0, 1<<4],
							  [1<<1, 1<<2, 1<<3]])

	
	def __init__(self):
		# Vectorizar funciones
		#self.__vect_compute_neigh_value = np.vectorize(
		#										LBPDescriptor._compute_neigh_value,
		#											  excluded=[0])

		self.__vect_calc_cell_hist = np.vectorize(self.__calc_cell_hist,
													 excluded=[0],
													 signature='(),()->(n)')

	def __do_cartesian_product(a: np.array, b: np.array):
		return np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])


	# Determinar todas las combinaciones de índices i,j por las que se va
	# a aplicar la operación de vecindad
	__CELL_INDEXES = np.arange(1, __CELL_SIZE+1)
	__CELL_INDEXES = __do_cartesian_product(__CELL_INDEXES, __CELL_INDEXES)

	def _compute_neigh_value(arr, i, j):

		# Determinar los valores del grid mayores que su píxel central
		#lbp_mask = np.where(grid >= grid[1,1], 1, 0)
		lbp_mask = np.where(arr[i-1:i+2,
								  j-1:j+2] >= arr[i,j], 1, 0)

		# Determinar valor asociado al grid y devolverlo
		return (lbp_mask*LBPDescriptor.__MASK_CELL).sum()
	
	_vect_compute_neigh_value = np.vectorize(_compute_neigh_value, excluded=[0])
	
	def __calc_cell_hist(self, img, i_start, j_start):
		
		
		# Celda auxiliar que facilitará el cálculo en posiciones fuera de la
		# imagen, que se considerará por defecto que tienen valor 0
		cell = np.zeros((LBPDescriptor.__CELL_SIZE + 2,
						   LBPDescriptor.__CELL_SIZE + 2),
						dtype=img.dtype)

		# Copiar celda original en el centro de esta auxiliar, los bordes
		# tendrán valor 0
		cell[1:-1, 1:-1] = img[i_start:(i_start + LBPDescriptor.__CELL_SIZE),
								  j_start:(j_start + LBPDescriptor.__CELL_SIZE)]
	
		# Se computa el cálculo de la rejilla LBP a toda la región usando
		# el siguiente procedimiento vectorizado:
		
		# for i in range(1,self.__CELL_SIZE+1):
		#	 for j in range(1,self.__CELL_SIZE+1):
		#	    	lbp_matrix[i,j] = self.__compute_neigh_value(cell[i-1:i+1,
		#																  j-1:j+1])
	
		# Determinar todas las combinaciones de índices i,j por las que se va
		# a aplicar la operación
		#i = np.arange(1, LBPDescriptor.__CELL_SIZE+1)
		#indexes = LBPDescriptor.__do_cartesian_product(i, i)
		

		# Aplicar cálculo de la rejilla LBP a toda la matriz
		lbp_matrix = self._vect_compute_neigh_value(cell,
									  i=self.__CELL_INDEXES[:,1],
									   j=self.__CELL_INDEXES[:,0]).astype(np.uint8)
		#reshape((len(i), len(i)))
	
		# Calcular histograma de la matriz
		hist = np.histogram(lbp_matrix, bins=256, density=True,
					  range=(0, 256))[0].astype(np.float32)
	
		return hist
	
	def __compute_LBP_descriptors(self, img):
		
		img_height, img_width = img.shape
		
		# Preparar iteradores
		it_i = np.arange(0, img_height - LBPDescriptor.__V_SHIFT,
				   LBPDescriptor.__V_SHIFT)
		it_j = np.arange(0, img_width - LBPDescriptor.__H_SHIFT,
				   LBPDescriptor.__H_SHIFT)
		
		it = LBPDescriptor.__do_cartesian_product(it_j, it_i)

		#it_i = zip(
		#		range(0, img_height, self._V_SHIFT),
		#		range(self._WIND_SIZE -1, img_height, self._V_SHIFT)
		#	)
		
		#it_j = zip(
		#		range(0, img_width, self._H_SHIFT),
		#		range(self._WIND_SIZE -1, img_width, self._H_SHIFT)
		#	)

		# Procesar por ventanas mediante un procedimiento vectorizado del
		# siguiente código:
		
		# for i in it_i:
		#	for j in it_j:
		#		desc[i,j] = self.___compute_LBP_cell_hist(img, i, j)
		desc = self.__vect_calc_cell_hist(img, it[:,1], it[:,0]).flatten()
		
		return desc

	def compute(self, img):
		
		# Convertir la imagen en escala de grises
		gray = cvtColor(img, COLOR_BGR2GRAY)
		
		return self.__compute_LBP_descriptors(gray)

class UniformLBPDescriptor(LBPDescriptor):
	
	def generate_uniforms(n_bits):
		
		"""Función que genera todos los uniformes considerando n_bits,
			considerando como no uniforme cualquier valor numérico
			con un número de transiciones cíclicas entre 0 y 1 de 0 o 2
			transiciones.
		"""
		
		bit_mask = (1<<(n_bits)) - 1 # Máscara con n_bits adyacentes a 1
		uniforms = [0, bit_mask]
		
		for i in range(1,n_bits):
			# Base a partir de la cual se generan otros uniformes mediante
			# desplazamientos bit a bit circulares
			base = (1 << i) - 1
			uniforms.append(base)

			for j in range(1,n_bits):
				aux = base << j
				
				# Trasladar los bits mayores posteriores a n_bits a los bits
				# menos significativos puesto que el desplazamiento es circular
				# (el bit más significativo es proseguido por el bit menos
				# significativo formando un ciclo)
				aux |= (aux & ~bit_mask) >> n_bits
				
				# Eliminar los bits trasladados
				aux &= bit_mask

				uniforms.append(aux)
		
		# Finalmente se devuelven los uniformes ordenados de menor a mayor
		uniforms.sort()
		
		return uniforms
	
	# Generar todos los uniformes existentes entre 0 y 255
	__UNIF_CODES = generate_uniforms(n_bits=8)
	__UNIF_CODES = dict(zip(__UNIF_CODES, range(len(__UNIF_CODES))))
	
	def _compute_neigh_value(arr, i, j):
		
		# Computar el valor de vecindad cásico de LBPDescriptor
		neigh_value = LBPDescriptor._compute_neigh_value(arr,i,j)
		
		# Asignar su valor 
		return (UniformLBPDescriptor.__UNIF_CODES[neigh_value] if neigh_value
				  in UniformLBPDescriptor.__UNIF_CODES else
					  len(UniformLBPDescriptor.__UNIF_CODES))

	_vect_compute_neigh_value = np.vectorize(_compute_neigh_value, excluded=[0])
	
	def __init__(self):
		super().__init__()
		
		# Vectorizar la función redefinida
		#self.__vect_compute_neigh_value = np.vectorize(
		#								UniformLBPDescriptor._compute_neigh_value,
		#									  excluded=[0])
