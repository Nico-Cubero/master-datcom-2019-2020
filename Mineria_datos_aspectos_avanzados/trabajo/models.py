# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 02:54:20 2020

@author: nico
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, Dense, MaxPooling2D, Dropout,
					Flatten, GlobalAveragePooling2D, GaussianNoise)
from tensorflow.keras.applications.vgg16 import VGG16

def model1():

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x16
	"""
	model.add(Conv2D(filters=16, kernel_size=2, input_shape=(256,256,3),
				  strides=(2,2), activation='relu'))
	
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x16
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x16
			32 filtros
			Output: 64x64x32
	"""
	model.add(Conv2D(filters=32, kernel_size=2, input_shape=(127,127,16),
			   strides=(2,2), activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 16x16x32
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 5: Convolucional 2D
			Input: Array de 64x64x32
			64 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=2, input_shape=(32,32,32),
			   strides=(2,2), activation='relu'))	
	
	"""
		Capa 5: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 8x8x64
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 6: Flatten
			Convertiría la salida a un vector de 8x8x64 = 4096 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 7: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(1024))
	model.add(Dropout(0.1))
	
	"""
		Capa 8: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model2():

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			16 filtros
			Output: 128x128x16
	"""
	model.add(Conv2D(filters=16, kernel_size=2, input_shape=(256,256,3),
			activation='relu'))

	"""
		Capa 6: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model3():

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x16
	"""
	model.add(Conv2D(filters=16, kernel_size=2, input_shape=(256,256,3),
				  strides=(2,2), activation='relu'))
	
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x16
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x16
			32 filtros
			Output: 64x64x32
	"""
	model.add(Conv2D(filters=32, kernel_size=2, input_shape=(127,127,16),
			   strides=(2,2), activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 16x16x32
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 5: Convolucional 2D
			Input: Array de 64x64x32
			64 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=2, input_shape=(32,32,32),
			   strides=(2,2), activation='relu'))	
	
	"""
		Capa 5: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 8x8x64
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 6: Flatten
			Convertiría la salida a un vector de 8x8x64 = 4096 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 7: Densa 
			Capa con 1024 entradas
			Dropout del 20 %
	"""
	model.add(Dense(1024))
	model.add(Dropout(0.2))

	"""
		Capa 8: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(512))
	model.add(Dropout(0.1))
	
	"""
		Capa 9: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model4():

	"""
		Se parte del model 2, pero se trata de no reducir tanta información en
		las capas de convolución
	"""
	
	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 255x255x16
	"""
	model.add(Conv2D(filters=16, kernel_size=7, input_shape=(256,256,3),
				       activation='relu'))
	
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 128x128x16
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 128x128x16
			32 filtros
			Output: 127x127x32
	"""
	model.add(Conv2D(filters=32, kernel_size=5, activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x32
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 64x64x32
			32 filtros
			Output: 63x63x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 32x32x64
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 7: Convolucional 2D
			Input: Array de 29x29x32
			32 filtros
			Output: 14x14x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))

	
	"""
		Capa 8: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 7x7x128
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 5: Flatten
			Convertiría la salida a un vector de 7x7x128 = 6272 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 6: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(1024))
	model.add(Dropout(0.1))
	
	"""
		Capa 7: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

"""
def model5():

	"" "
		Se parte del model 4, y se prueba a añadir otra capa densa más
	"" "
	
	model = Sequential()
	
	"" "
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 255x255x16
	"" "
	model.add(Conv2D(filters=16, kernel_size=7, input_shape=(256,256,3),
				       activation='relu'))
	
	"" "
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 128x128x16
	"" "
	model.add(MaxPooling2D(pool_size=2))
	
	"" "
		Capa 3: Convolucional 2D
			Input: Array de 128x128x16
			32 filtros
			Output: 127x127x32
	"" "
	model.add(Conv2D(filters=32, kernel_size=5, activation='relu'))
	
	"" "
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x32
	"" "
	model.add(MaxPooling2D(pool_size=2))

	"" "
		Capa 5: Convolucional 2D
			Input: Array de 64x64x32
			32 filtros
			Output: 63x63x64
	"" "
	model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))

	
	"" "
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 32x32x64
	"" "
	model.add(MaxPooling2D(pool_size=2))

	"" "
		Capa 7: Convolucional 2D
			Input: Array de 29x29x32
			32 filtros
			Output: 14x14x128
	"" "
	model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))

	
	"" "
		Capa 8: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 7x7x128
	"" "
	model.add(MaxPooling2D(pool_size=2))
	
	"" "
		Capa 5: Flatten
			Convertiría la salida a un vector de 7x7x128 = 6272 dimensiones
	"" "
	model.add(Flatten())
	
	"" "
		Capa 6: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"" "
	model.add(Dense(1024))
	model.add(Dropout(0.2))
	
	"" "
		Capa 7: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"" "
	model.add(Dense(512))
	model.add(Dropout(0.1))

	"" "
		Capa 7: Softmax de clasificación
			Capa de 54 entradas
	"" "
	model.add(Dense(45, activation='softmax'))
	
	return model
"""

def model5():

	"""
		Se parte del model 4, y se prueba a añadir otra capa densa más
	"""
	
	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 255x255x16
	"""
	model.add(Conv2D(filters=8, kernel_size=7, strides=(2,2), input_shape=(256,256,3),
				       activation='relu'))
	
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 128x128x16
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 128x128x16
			32 filtros
			Output: 127x127x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2), activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x32
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 64x64x32
			32 filtros
			Output: 63x63x64
	"""
	model.add(Conv2D(filters=32, kernel_size=2, strides=(2,2), activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 32x32x64
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 5: Flatten
			Convertiría la salida a un vector de 7x7x128 = 6272 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 6: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(1024))
	model.add(Dropout(0.2))
	
	"""
		Capa 7: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(512))
	model.add(Dropout(0.1))

	"""
		Capa 7: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model6():

	"""
		El modelo 2 muestra infraentrenamiento, mientras que los modelos
		4 y 5 muestran algo de sobreaprendizaje. En todos los casos, se
		aprecia que el modelo no ajusta correctamente y muestra un bajo
		rendimiento sobre el conjunto de validación, pero con los 
		modelos 4 y 5 parece que el número de capas es excesivo
		(experimentos 5 y 6).

		Se pretende elaborar un modelo algo más complejo que el modelo 2
		sin llegar a la complejidad de los modelos 4 y 5 y a aumentar
		el tamaño del kernel, parámetro influyente que hasta ahora no
		se había tocado demasiado.

		Se plantea incrementar también el número de filtros, no obstante,
		quizás halla que incrementarlos más.

		Habría que revisar por tanto, los tamaños de kernel, la conveniencia
		o no de usar strides de 2 en TODAS las capas convolucionales
		(esto podría eliminar parte de la información si se abusa
		demasiado), el número de filtros (quizás sean necesarios más
		filtros) y el número de capas densas así como su longitud
	"""
	
	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 127x127x32
	"""
	model.add(Conv2D(filters=32, kernel_size=15, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	# No se ponen 15 filtros porque revienta la memoria
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x32
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x32
			64 filtros
			Output: 32x32x64
	"""
	model.add(Conv2D(filters=64, kernel_size=9, strides=(2,2), activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 16x16x64
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			32 filtros
			Output: 8x8x92
	"""
	model.add(Conv2D(filters=92, kernel_size=5, strides=(2,2), activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 4x4x92
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 7: Convolucional 2D
			Input: Array de 4x4x128
			128 filtros
			Output: 8x8x92
	"""
	#model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 4x4x92
	"""
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Flatten
			Convertiría la salida a un vector de 4x4x92 = 1472 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 8: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(512))
	model.add(Dropout(0.1))
	

	"""
		Capa 9: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model7():

	"""
		Se parte del modelo 6, pero se reducen a la mitad el número
		de filtros
	"""
	
	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 127x127x32
	"""
	model.add(Conv2D(filters=16, kernel_size=15, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	# No se ponen 15 filtros porque revienta la memoria
	"""
		Capa 2: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 64x64x32
	"""
	model.add(MaxPooling2D(pool_size=2))
	
	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x16
			64 filtros
			Output: 32x32x64
	"""
	model.add(Conv2D(filters=32, kernel_size=9, strides=(2,2), activation='relu'))
	
	"""
		Capa 4: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 16x16x64
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x46
			32 filtros
			Output: 8x8x92
	"""
	model.add(Conv2D(filters=46, kernel_size=5, strides=(2,2), activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 4x4x46
	"""
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 7: Convolucional 2D
			Input: Array de 4x4x128
			128 filtros
			Output: 8x8x92
	"""
	#model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))

	
	"""
		Capa 6: Max Polling 2D
			Reduciría la dimensión de la anterior salida a 4x4x92
	"""
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Flatten
			Convertiría la salida a un vector de 4x4x92 = 1472 dimensiones
	"""
	model.add(Flatten())
	
	"""
		Capa 8: Densa 
			Capa con 512 entradas
			Dropout del 10 %
	"""
	model.add(Dense(512))
	model.add(Dropout(0.1))
	

	"""
		Capa 9: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model_cfar10():

	"""
		Probar con el modelo usado para el cfar10
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, input_shape=(256,256,1),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, input_shape=(256,256,1),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, input_shape=(256,256,1),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model8():

	"""
		Parte del modelo cfar0, pero añade un stride de 2 en la primera
		y segunda capas convolucionales para tratar de reducir el
		sobreentrenamiento
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model9():

	"""
		Parte del modelo 8 (basado en elde cfar), pero elimina otra capa
		convolucional más para eliminar el sobreentrenamiento. Si
		infraaprende, subir la capa Densa de 64 nodos. Se añade también
		a este un Dropout del 10%
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	


	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model10():

	"""
		Parte del modelo 9, pero se le elimina la capa densa intermedia
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	


	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model11():

	"""
		Parte del modelo 10, pero se reduce a la mitad el número de filtros
		de las capas convolucionales, ya que se cree que puedan ser las
		causantes del sobreajuste
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	


	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model12():

	"""
		Parte del modelo 11, pero se elimina otra capa convolucional,
		Se quiere llevar al modelo al infraaprendizaje, ya que la
		tendencia que se observa es rara (quitándo capas sobreentrena
		más).
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))


	"""
		Capa 2: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 3: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model13():

	"""
		Parte del modelo 8, pero añade un stride de 2 a la 3a capa
		convolucional y aumenta el número de filtros a 96.
		Y añade un dropout del 10% a la capa densa intermedia
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model14():

	"""
		Parte del modelo 3, pero se incrementan los filtros de cada capa
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=128, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model15():

	"""
		Parte del modelo 15, pero añade otra capa convolucional más
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=128, kernel_size=2, strides=(2,2),
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model16():

	"""
		Parte del modelo 15, pero se elimina el stride 2 en la última
		capa convolucional
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model17():

	"""
		Parte del modelo 16, pero se ajusta el tamaño del kernel en la
		tercera capa a 2 y se disminuye el número de filtros de la última
		capa de 128 a 96
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model18():

	"""
		Parte del modelo 17, pero se aumenta el tamaño de kernel de la
		primera capa a 5. También se incrementa el número de filtros
		de la última capa a 112
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=112, kernel_size=2,
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model19():

	"""
		Parte del modelo 18, pero se decrementa el número de filtros
		de la última capa convolucional a 96
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model20():

	"""
		Parte del modelo 19, pero elimina la capa Densa intermedia por
		ver si no sobreentrena
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=5, strides=(2,2),
			input_shape=(256,256,1), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=64, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	#model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model21():

	"""
		Parte del modelo 13, pero reduce a la mitad los filtros de las
		capas convolucionales
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=48, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model22():

	"""
		Parte del modelo 21, pero reduce a la mitad los filtros de las
		capas convolucionales
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model23():

	"""
		Parte del modelo 22, pero se aumenta el número de nodos de la
		capa Densa intermedia a 96
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model24():

	"""
		Parte del modelo 23, pero se aumenta el número de nodos de la
		capa Densa intermedia a 128 y se le añade un Dropout del 15%
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 128 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.15))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model25():

	"""
		Parte del modelo 23, pero se decrementa el número de filtros de la 3a
		convolucional a 16 y se aumenta el Dropout de la capa densa intermedia
		al 20 %
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 256x256x32
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model26():

	"""
		Parte del modelo 25, pero incrementa el número de nodos de la capa densa
		a 112
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(112, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model27():

	"""
		Parte del modelo 26, pero se añade otra capa Densa
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model28():

	"""
		Parte del modelo 27, pero se aumenta el número de nodos de la primera
		capa densa intermedia
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model29():

	"""
		Parte del modelo 27, pero se incrementa el número de filtros
		de la 3a capa convolucional
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model30():

	"""
		Parte del modelo 29, pero se incrementa el número de filtros
		de todas las capas en un 25%
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=10, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=20, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model31():

	"""
		Parte del modelo 29, pero se decrementa el número de filtros en un 25%
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=6, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=12, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model32():

	"""
		Parte del modelo 29, pero se incrementa el número de filtros
		de las capas convolucionales:
		(1a capa: 1.5, 2a capa: 1.25, 3a capa: 1.1)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=12, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=20, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=36, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model33():

	"""
		Parte del modelo 29, pero se incrementa el número de filtros
		de las capas convolucionales (siguiendo la idea del modelo 32):
		(1a capa: 1.75, 2a capa: 1.5, 3a capa: 1.25)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model34():

	"""
		Parte del modelo 29, pero se incrementa el número de filtros
		de las capas convolucionales (siguiendo la idea del modelo 32):
		(1a capa: 2, 2a capa: 1.75, 3a capa: 1.5)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=28, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model35():

	"""
		Parte del modelo 33, pero se añade otra capa convolucional más
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model36():

	"""
		Parte del modelo 35, pero se elimina MaxPooling de la última capa
		convolucional y se incrementa el número de capas de la primera capa
		densa a 128
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model37():

	"""
		Parte del modelo 36, pero incrementa el número de nodos de las
		capas densas en 1.5 para la 1a y 1.25 para la 2a, además de
		incrementar el Dropout de la 1a a 0.2
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(192, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model38():

	"""
		Parte del modelo 36, pero incrementa el número de la 1a capa
		densa a 256 y el de la 2a capa densa a 96
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model39():

	"""
		Parte del modelo 37, decrementa el número de nodos de la
		primera capa convolucional a 10 filtros, la de la 2a a 20
		filtros y la 3a a 32 filtros
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=10, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=20, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=48, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(192, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model40():

	"""
		Parte del modelo 37, pero se decrementa el número de nodos de la
		1a capa convolucional a 8 filtros, la de la 2a a 16, la de la
		3a a 24 y la de la 4a a 32
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=8, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x32
	"""
	model.add(Conv2D(filters=32, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(192, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model41():

	"""
		Parte del modelo 37, pero incrementa el número de nodos de la
		última capa convolucional a 52
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(192, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model42():

	"""
		Parte del modelo 41, pero incrementa el número de nodos de las
		capas densas intermedias (1a a 256, 2a 96)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model43():

	"""
		Parte del modelo 42, pero decrementa el número de nodos de las capas
		densas (1a capa: 96, 2a capa 64)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=40, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model44():

	"""
		Parte del modelo 43, pero incrementa el número de filtros de las capas
		convolucionales (1a capa 1.5, 2a capa 1.25, 3a 1.1) y además, se
		incrementa el tamaño del kernel de la 1a capa a 5
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=5, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=44, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model45():

	"""
		Parte del modelo 44, pero se elimina el stride de la capa convolucional
		3 y se añade otra capa convolucional más
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=5, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 128x128x16 = 4096 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model46():

	"""
		Parte del modelo 45, pero se incrementa el número de nodos en las capas
		densas intermedias (1a capa a 128, 2a capa a 80)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=5, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 2x2x64
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x64 =  1024 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model47():

	"""
		Parte del modelo 46, pero se incrementa el número de nodos en las capas
		densas intermedias (1a capa a 256, 2a capa a 128)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=5, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 2x2x64
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x64 =  1024 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model48():

	"""
		Parte del modelo 47, pero se incrementa el número de nodos en las capas
		densas intermedias (1a capa a 512, 2a capa a 128)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 64x64x8
	"""
	model.add(Conv2D(filters=14, kernel_size=5, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 64x64x8
			32 filtros
			Output: 16x16x16
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))	

	"""
		Capa 3: Convolucional 2D
			Input: Array de 16x16x16
			32 filtros
			Output: 4x4x16
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 4x4x16
			32 filtros
			Output: 2x2x64
	"""
	model.add(Conv2D(filters=64, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x64 =  1024 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model49():

	"""
		Parte del modelo 45, pero reestructuran las capas convolucionales
		eliminando los MaxPooling de las 2 primeras capas convolucionales y
		se añaden otras 2 capas convolucionales más
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model50():

	"""
		Parte del modelo 49, pero se incrementan el número de nodos de
		las capas densas (1a capa: 128, 2a capa: 96)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model51():

	"""
		Parte del modelo 50, pero se decrementa el número de nodos de
		la 2a capa densa a 64
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model52():

	"""
		Parte del modelo 51, pero se decrementa el número de nodos de
		la 1a capa densa a 86
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(86, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model53():

	"""
		Parte del modelo 52, pero se elimina la 1a capa densa
		y la 2a se pone con 80 nodos
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model54():

	"""
		Parte del modelo 53, pero se incrementa el número de nodos de
		la 1a capa Densa a 200.
		Experimento borrado
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model55():

	"""
		Parte del modelo 51, pero se añade otra capa densa con 80 nodos
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=14, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=30, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=44, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x64
	"""
	model.add(Conv2D(filters=96, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model56():

	"""
		Parte del modelo 51, pero se incrementa el número de filtros
		de las capas convolucionales
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x128 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model57():

	"""
		Parte del modelo 56, pero se elimina una capa convolucional
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 8x8x128 = 8192 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model58():

	"""
		Parte del modelo 56, pero se decrementa el número de filtros
		de 1a capa densa
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model59():

	"""
		Parte del modelo 58, pero se decrementa el número de filtros
		de 1as capas densas (1a capa a 80, 2a capa a 54)
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(54, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model60():

	"""
		Parte del modelo 58, pero se incrementa el número de nodos de la 2a
		capa densa a 80
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model61():

	"""
		Parte del modelo 60, pero añade otra capa densa
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(96, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(80, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model62():

	"""
		Parte del modelo 58, pero se decrementa el número de filtros
		de 1a 1a capa densa a 70 y establece un batch_size de 1 en la última
		capa convolucional
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model63():

	"""
		Parte del modelo 62, pero establece el batch_size de la 4a capa
		convolucional a 1.
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=16, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model64():

	"""
		Parte del modelo 62, pero se incrementa el número de filtros de
		las capas convolucionales
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=24, kernel_size=3, strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=48, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=64, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=128, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=254, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=512, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model65_conv2d_filters(params=dict()):

	# Establecer un valor por defecto al número de filtros de la 1a capa conv
	if 'conv2d_1_filters' not in params:
		params['conv2d_1_filters'] = 16

	if 'conv2d_2_filters' not in params:
		params['conv2d_2_filters'] = 32

	if 'conv2d_3_filters' not in params:
		params['conv2d_3_filters'] = 54

	"""
		Parte del modelo 63, pero admite parametrización en el número
		de filtros de la 1a capa convolucional
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=params['conv2d_1_filters'], kernel_size=3,
			strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=params['conv2d_2_filters'], kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=params['conv2d_3_filters'], kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96, kernel_size=3,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(Conv2D(filters=254, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model66_conv2d_filters(params=dict()):

	# Establecer un valor por defecto al número de filtros de la 1a capa conv
	if 'conv2d_1_filters' not in params:
		params['conv2d_1_filters'] = 16

	if 'conv2d_2_filters' not in params:
		params['conv2d_2_filters'] = 32

	if 'conv2d_3_filters' not in params:
		params['conv2d_3_filters'] = 54

	if 'conv2d_4_filters' not in params:
		params['conv2d_4_filters'] = 96

	"""
		Parte del modelo 65, pero admite parametrización en el número
		de filtros de las capas convolucionales, establece un stride
		en la 3a capa convolucional por problemas de memoria
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=params['conv2d_1_filters'], kernel_size=3,
			strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=params['conv2d_2_filters'], kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=params['conv2d_3_filters'], kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	if 'conv2d_4_filters' in params:
		model.add(Conv2D(filters=params['conv2d_4_filters'],
				kernel_size=3, activation='relu'))
		model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())

	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.2))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model67_sep_conv_dense(conv_conf: dict, dense_conf=None):

	"""
		Parte del modelo 65, pero admite parametrización tanto en la estructura
		de capas convolucionales como en la estructura de capas densas
	"""	
	model = Sequential()
	
	"""
		Capas convolucionales
		Entrada: 256x256x3
	"""

	model.add(Conv2D(filters=conv_conf['num_filters'][0],
					kernel_size=conv_conf['kernel_size'][0],
			strides=(conv_conf['stride'][0],)*2,
			input_shape=(256,256,3), activation='relu'))

	for i in range(1, len(conv_conf['num_filters'])):
		model.add(Conv2D(filters=conv_conf['num_filters'][i],
						kernel_size=conv_conf['kernel_size'][i],
				strides=(conv_conf['stride'][i],)*2,
				activation='relu'))

		# Añadir capa de Max Pooling
		if conv_conf['max_pooling'][i]:
			model.add(MaxPooling2D(pool_size=2))


	"""
		Capa : Flatten
			Convertiría la de la convolución a un vector
	"""
	model.add(Flatten())


	# Añadir capas densas intermedias si se han especificado
	"""
		Capas densas intermedias
	"""

	if dense_conf:
		for i in range(len(dense_conf['layers'])):
			model.add(Dense(dense_conf['layers'][i], activation='relu'))
			model.add(Dropout(dense_conf['dropout'][i]))

	"""
		Capa Softmax de clasificación
			Capa de 45 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model

def model_transfer_vgg16():

	"""
		Modelo base VGG16
	"""
	base_model = VGG16(include_top=False, weights='imagenet',
			input_shape=(256, 256, 3))

	base_model.trainable = False

	"""
		Capas Densas extra

		Capa Densa 1: 70 unidades

		Capa Densa 2: 64 unidades

		Capa Softmax: 45 unidades
	"""

	x = GlobalAveragePooling2D()(base_model.output)

	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.1)(x)

	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)

	x = Dense(45, activation='softmax')(x)

	# Conformar el modelo completo
	model = Model(inputs=base_model.input, outputs=x)

	return model

def model68_def():

	"""
		El mejor modelo encontrado
	"""	

	model = Sequential()
	
	"""
		Capa 1: Convolucional 2D
			Input: Array de 256x256
			32 filtros
			Output: 128x128x14
	"""
	model.add(Conv2D(filters=20, kernel_size=3,
			strides=(2,2),
			input_shape=(256,256,3), activation='relu'))

	"""
		Capa 2: Convolucional 2D
			Input: Array de 128x128x14
			32 filtros
			Output: 64x64x30
	"""
	model.add(Conv2D(filters=32, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 3: Convolucional 2D
			Input: Array de 64x64x30
			32 filtros
			Output: 32x32x44
	"""
	model.add(Conv2D(filters=54, kernel_size=3, strides=(2,2),
			activation='relu'))

	"""
		Capa 4: Convolucional 2D
			Input: Array de 32x32x44
			32 filtros
			Output: 16x16x64
	"""
	model.add(Conv2D(filters=96,
			kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 5: Convolucional 2D
			Input: Array de 16x16x64
			64 filtros
			Output: 8x8x96
	"""
	model.add(Conv2D(filters=128, kernel_size=2,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 6: Convolucional 2D
			Input: Array de 8x8x64
			64 filtros
			Output: 4x4x128
	"""
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(filters=254, kernel_size=1,
			activation='relu'))
	model.add(MaxPooling2D(pool_size=2))

	"""
		Capa 4: Flatten
			Convertiría la salida a un vector de 4x4x256 = 2048 dimensiones
	"""
	model.add(Flatten())
	model.add(GaussianNoise(0.1))
	"""
		Capa 2: Softmax de clasificación
			Capa de 70 entradas
	"""
	model.add(Dense(70, activation='relu'))
	model.add(Dropout(0.15))

	"""
		Capa 2: Softmax de clasificación
			Capa de 64 entradas
	"""
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))

	"""
		Capa 2: Softmax de clasificación
			Capa de 54 entradas
	"""
	model.add(Dense(45, activation='softmax'))
	
	return model



model = {
			'model1': model1,
			'model2': model2,
			'model3': model3,
			'model4': model4,
			'model5': model5,
			'model6': model6,
			'model7': model7,
			'model_cfar10': model_cfar10,
			'model8': model8,
			'model9': model9,
			'model10': model10,
			'model11': model11,
			'model12': model12,
			'model13': model13,
			'model14': model14,
			'model15': model15,
			'model16': model16,
			'model17': model17,
			'model18': model18,
			'model19': model19,
			'model20': model20,
			'model21': model21,
			'model22': model22,
			'model23': model23,
			'model24': model24,
			'model25': model25,
			'model26': model26,
			'model27': model27,
			'model28': model28,
			'model29': model29,
			'model30': model30,
			'model31': model31,
			'model32': model32,
			'model33': model33,
			'model34': model34,
			'model35': model35,
			'model36': model36,
			'model37': model37,
			'model38': model38,
			'model39': model39,
			'model40': model40,
			'model41': model41,
			'model42': model42,
			'model43': model43,
			'model44': model44,
			'model45': model45,
			'model46': model46,
			'model47': model47,
			'model48': model48,
			'model49': model49,
			'model50': model50,
			'model51': model51,
			'model52': model52,
			'model53': model53,
			'model54': model54,
			'model55': model55,
			'model56': model56,
			'model57': model57,
			'model58': model58,
			'model59': model59,
			'model60': model60,
			'model61': model61,
			'model62': model62,
			'model63': model63,
			'model64': model64,
			'model65_conv2d_filters': model65_conv2d_filters,
			'model66_conv2d_filters': model66_conv2d_filters,
			'model67_sep_conv_dense': model67_sep_conv_dense,
			'model_transfer_vgg16': model_transfer_vgg16,
			'model68_def': model68_def
		}

