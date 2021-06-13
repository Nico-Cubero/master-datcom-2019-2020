# -*- coding: utf-8 -*-
"""
@author: nico
"""
import json
import numpy as np
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import (Conv2D, Dense, MaxPooling2D, Dropout,
					Flatten, Activation, GaussianNoise, ZeroPadding2D,
					BatchNormalization, Input, Lambda, Conv2DTranspose)
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2


"""
def AlexNet(weights=None, input_shape=(227, 227, 3), include_top=True):

	""
		Tomado de https://www.mydatahack.com/building-alexnet-with-keras/
	""

	model = Sequential()

	model.add(Lambda(lambda image: keras.backend.resize_images(image, (227, 227))))

	model.add(Input(shape=(227,227,3)))

	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4),
		padding='valid', name='conv1'))
	model.add(Activation('relu'))

	# Pooling
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

	# Batch Normalisation before passing it to the next layer
	model.add(BatchNormalization())

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1),
		padding='valid', name='conv2'))
	model.add(Activation('relu'))

	# Pooling
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

	# Batch Normalisation
	model.add(BatchNormalization())

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='valid', name='conv3'))
	model.add(Activation('relu'))

	# Batch Normalisation
	model.add(BatchNormalization())

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1),
		padding='valid', name='conv4'))
	model.add(Activation('relu'))

	# Batch Normalisation
	model.add(BatchNormalization())

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1),
		padding='valid', name='conv5'))
	model.add(Activation('relu'))

	# Pooling
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

	# Batch Normalisation
	model.add(BatchNormalization())

	# Passing it to a dense layer
	model.add(Flatten())

	# 1st Dense Layer
	model.add(Dense(4096, name='fc6')) #input_shape=(224*224*3,)
	model.add(Activation('relu'))

	# Add Dropout to prevent overfitting
	model.add(Dropout(0.2))

	# Batch Normalisation
	model.add(BatchNormalization())

	# 2nd Dense Layer
	model.add(Dense(4096, name='fc7'))
	model.add(Activation('relu'))

	# Add Dropout
	model.add(Dropout(0.4))

	# Batch Normalisation
	model.add(BatchNormalization())

	#if include_top:
	# 3rd Dense Layer
	model.add(Dense(1000, name='fc8'))
	model.add(Activation('softmax'))

	if weights:
		w_data = np.load('bvlc_alexnet.npy', encoding='bytes', allow_pickle=True).item()

		for l in w_data:
			print(l)
			model.get_layer(name=l).set_weights(w_data[l])

	if not include_top:
		# Eliminar las 2 últimas capas si se decide no incluirlas
		model.pop()
		model.pop()

	# Reemplazar capa de entrada por la especificada con el tamaño
	#model.layers.pop(0)
	#input = Input(shape=input_shape)
	#output = model(input)

	#model = Model(input, output)

	return model
"""

def model_transfer_vgg16(end_layers: dict or str=None, load_weights=True):

	"""
		Modelo base VGG16
	"""
	base_model = VGG16(include_top=False, weights='imagenet' if load_weights else None,
			input_shape=(128, 64, 3))

	base_model.trainable = not load_weights

	"""
		Añadir capas de terminación
	"""

	if end_layers:
		# Cargar las capas de terminación y añadirlas al modelo inicial
		end_model = model_from_json(json.dumps(end_layers) if isinstance(end_layers, dict) else end_layers)

		x = end_model.layers[0](base_model.output)

		for i in range(1, len(end_model.layers)):
			x = end_model.layers[i] (x)

		#end_model.layers[0] = end_model.layers[0](base_model.output)

		x = Dense(1, activation='sigmoid')(x) #len(end_model.layers)
	else:
		x = Dense(1, activation='sigmoid')(base_model.output)

	# Construir arquitectura entera
	model = Model(inputs=base_model.input, outputs=x)

	return (model, base_model, end_model)

"""
def model_transfer_alexnet(end_layers: dict or str=None, load_weights=True):

	""
		Modelo base VGG16
	""
	base_model = AlexNet(weights="./utils/alexnet_weights.h5" if load_weights else None,
							input_shape=(128,64,3), include_top=False)

	base_model.trainable = not load_weights

	# De este modelo se debe de suprimir la última capa Softmax (implementadas
	# como una capa densa y otra de activación)

	""
		Añadir capas de terminación
	""

	if end_layers:
		# Cargar las capas de terminación y añadirlas al modelo inicial
		end_model = model_from_json(json.dumps(end_layers) if isinstance(end_layers, dict) else end_layers)

		x = end_model.layers[0](base_model.layers[-2])

		for i in range(1, len(end_model.layers)-2):
			x = end_model.layers[i] (x)

		#end_model.layers[0] = end_model.layers[0](base_model.output)

		x = Dense(1, activation='sigmoid')(x) #len(end_model.layers)
	else:
		x = Dense(1, activation='sigmoid')(base_model.layers[-2])

	# Construir arquitectura entera
	model = Model(inputs=base_model.input, outputs=x)

	return (model, base_model, end_model)
"""

"""
def model_transfer_ResNet18(end_layers: dict or str=None, load_weights=True):

	""
		Modelo base ResNet18
	""
	base_model = ResNet18(include_top=False, weights='imagenet' if load_weights else None,
			input_shape=(128, 64, 3))
	base_model = Model(base_model.input, base_model.output)

	base_model.trainable = not load_weights

		Añadir capas de terminación


	if end_layers:
		# Cargar las capas de terminación y añadirlas al modelo inicial
		end_model = model_from_json(json.dumps(end_layers) if isinstance(end_layers, dict) else end_layers)
		x = end_model.layers[0](base_model.output)

		for i in range(1, len(end_model.layers)):
			x = end_model.layers[i](x)

		#end_model.layers[0] = end_model.layers[0](base_model.output)

		x = Dense(1, activation='sigmoid')(x) #len(end_model.layers)
	else:
		x = Dense(1, activation='sigmoid')(base_model.output)

	# Construir arquitectura entera
	model = Model(inputs=base_model.input, outputs=x)

	return (model, base_model, end_model)
"""
def model_transfer_ResNet50(end_layers: dict or str=None, load_weights=True):

	"""
		Modelo base ResNet50
	"""
	base_model = ResNet50(include_top=False, weights='imagenet' if load_weights else None,
			input_shape=(128, 64, 3))

	base_model.trainable = not load_weights

	"""
		Añadir capas de terminación
	"""

	if end_layers:
		# Cargar las capas de terminación y añadirlas al modelo inicial
		end_model = model_from_json(json.dumps(end_layers) if isinstance(end_layers, dict) else end_layers)
		x = end_model.layers[0](base_model.output)

		for i in range(1, len(end_model.layers)):
			x = end_model.layers[i](x)

		#end_model.layers[0] = end_model.layers[0](base_model.output)

		x = Dense(1, activation='sigmoid')(x) #len(end_model.layers)
	else:
		x = Dense(1, activation='sigmoid')(base_model.output)

	# Construir arquitectura entera
	model = Model(inputs=base_model.input, outputs=x)

	return (model, base_model, end_model)

def model_transfer_MobileNetV2(end_layers: dict or str=None, load_weights=True):

	"""
		Modelo base MobileNetV2
	"""
	base_model = MobileNetV2(include_top=False, weights='imagenet' if load_weights else None,
			input_shape=(128, 64, 3), classes=2)

	base_model.trainable = not load_weights

	"""
		Añadir capas de terminación
	"""

	if end_layers:
		# Cargar las capas de terminación y añadirlas al modelo inicial
		end_model = model_from_json(json.dumps(end_layers) if isinstance(end_layers, dict) else end_layers)
		x = end_model.layers[0](base_model.output)

		for i in range(1, len(end_model.layers)):
			x = end_model.layers[i](x)

		#end_model.layers[0] = end_model.layers[0](base_model.output)

		x = Dense(1, activation='sigmoid')(x) #len(end_model.layers)
	else:
		x = Dense(1, activation='sigmoid')(base_model.output)

	# Construir arquitectura entera
	model = Model(inputs=base_model.input, outputs=x)

	return (model, base_model, end_model)

def model_segment():

	model = Sequential()

	model.add(Input(shape=(214, 214, 3)))

	# 1a capa Convolucional
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
		padding='same', name='conv1', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(ZeroPadding2D())

	# 2a capa Convolucional
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
		padding='same', name='conv2', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(ZeroPadding2D())

	# 3a capa Convolucional
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
		padding='same', name='conv3', activation='relu'))
	model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
		padding='same', name='conv4'))

	# 1a Deconvolución
	model.add(Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2),
		padding='same', name='transposed-conv-1'))

	# 2a Deconvolución
	model.add(Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2),
		name='transposed-conv-2'))
	model.add(ZeroPadding2D())

	# 4a Convolución
	model.add(Conv2D(filters=2, kernel_size=(1,1), strides=(1,1),
		name='conv5', activation='softmax'))

	return model
