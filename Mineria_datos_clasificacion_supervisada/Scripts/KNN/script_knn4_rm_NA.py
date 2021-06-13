# -*- coding: utf-8 -*-
"""
	@author Nicolás Cubero
"""

# Importación de librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack
import time
import json

#   Función para la ejecución de validación cruzada K-NN
def kfold_KNN(X, y, n_splits=5, k=5, jobs=-1, verbose=False):
    
    #   Métricas
    scores = []
    elapsed_time = []
    
    #   Realizar particionamiento
    i = 1
    kfold = StratifiedKFold(n_splits=n_splits)
    
    for train_index, test_index in kfold.split(X,y):
        
        #   Tomar el conjunto de train y test en esta partición
        X_train, Y_train = X[train_index], y[train_index]
        X_test, Y_test = X[test_index], y[test_index]

        t_start = time.time()
        
        #   Entrenar modelo sobre train
        if verbose:
            print('Entrenando knn con k={}, partición {} de {}'.format(k, i,
                                                                     n_splits))

        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=jobs)
        knn.fit(X_train, Y_train)
        
        #   Evaluar rendimiento del modelo
        score = knn.score(X_test, Y_test)
        t_end = time.time()

        #   Registrar métricas
        scores.append(score)
        elapsed_time.append(t_end-t_start)

        if verbose:
            print('Modelo evaluado con score={} en {} segundos'.format(
                  scores[-1], elapsed_time[-1]))
    
        i += 1
    
    #   Devolver resultado
    return {
             'avg_score': sum(scores)/len(scores),
             'min_score': min(scores),
             'max_score': max(scores),
             'avg_elapsed_time': sum(elapsed_time)/len(elapsed_time),
             'min_elapsed_time': min(elapsed_time),
             'max_elapsed_time': max(elapsed_time)
             }
        

# Lectura de datos
X_train = pd.read_csv('../train_values_4910797b-ee55-40a7-8668-10efd5c1b960.csv',
					sep=',', na_values=['','-'], index_col='id')
Y_train = pd.read_csv('../train_labels_0bf8bc6e-30d0-4c50-956a-603fc693d966.csv',
                    sep=',', na_values = ['','-'], index_col='id')

Y_train_values = Y_train.values.squeeze()

# Conjunto de test
X_test = pd.read_csv('../test_values_702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv',
                   sep=',', na_values = ['','-'], index_col='id')

index_test = X_test.index.values

"""
    Información de las variables
    
    - Categóricas: 'date_recorded', 'funder', 'installer', 'wpt_name', 'basin',
       'subvillage', 'region', 'lga', 'ward', 'public_meeting', 'recorded_by',
       'scheme_management', 'scheme_name', 'permit', 'extraction_type',
       'extraction_type_group', 'extraction_type_class', 'management',
       'management_group', 'payment', 'payment_type', 'water_quality',
       'quality_group', 'quantity', 'quantity_group', 'source', 'source_type',
       'source_class', 'waterpoint_type', 'waterpoint_type_group'
    - Enteras categóricas: region_code, district_code, num_private
    
    - Reales: amount_tsh, longitude, lattude
    - Enteras: gps_height, population,
                construction_year
                
    X_train.dtypes

"""

###############	Preprocesamiento	#################
droped_columns = ['date_recorded', 'wpt_name', 'recorded_by', 'scheme_name',
                  'management_group', 'payment_type', 'quality_group',
                  'quantity', 'source_class', 'waterpoint_type']


X_train = X_train.drop(columns=droped_columns)
X_test = X_test.drop(columns=droped_columns)

# Poner algunas variables enteras como categóricas
X_train[['region_code',
         'district_code',
         'num_private']] = X_train[['region_code', 'district_code',
                                   'num_private']].astype('object')
X_test[['region_code',
        'district_code',
        'num_private']] = X_test[['region_code', 'district_code',
                          'num_private']].astype('object')

# Sustituir valores inconsistentes por NAN
X_train['funder'] = X_train['funder'].replace('0', np.NaN)
X_train['installer'] = X_train['installer'].replace('0', np.NaN)
X_train['construction_year'] = X_train['construction_year'].replace('0', np.NaN)
X_train['latitude'] = X_train['latitude'].replace(-2e-08, np.NaN)
X_train['longitude'] = X_train['longitude'].replace(0, np.NaN)

X_test['funder'] = X_test['funder'].replace('0', np.NaN)
X_test['installer'] = X_test['installer'].replace('0', np.NaN)
X_test['construction_year'] = X_test['construction_year'].replace('0', np.NaN)
X_test['latitude'] = X_test['latitude'].replace(-2e-08, np.NaN)
X_test['longitude'] = X_test['longitude'].replace(0, np.NaN)

#   Eliminar filas con valores perdidos
index_train_na = X_train.isnull().any(axis=1)
index_test_na = X_test.isnull().any(axis=1)

X_train = X_train[~index_train_na]
Y_train_values = Y_train_values[~index_train_na] 
X_test = X_test[~index_test_na]

#   Almacenar los tipos de cada columna
dtypes = dict(zip(X_train.columns, X_train.dtypes))

# Separar variables numéricas y categóricas
X_train_cat = X_train[X_train.columns[(X_train.dtypes == 'object').values]]
X_train_num = X_train[X_train.columns[(X_train.dtypes != 'object').values]]

X_test_cat = X_test[X_test.columns[(X_test.dtypes == 'object').values]]
X_test_num = X_test[X_test.columns[(X_test.dtypes != 'object').values]]

# Binarizar variables categóricas
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train_cat)

X_train_cat = encoder.transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

#   Volver a unir conjunto variables categóricas y numéricas
X_train = hstack((X_train_cat, X_train_num)).tocsr()
X_test = hstack((X_test_cat, X_test_num)).tocsr()

#   Normalizar
normalizer = Normalizer()
normalizer.fit(X_train)

X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)


# Probar a hacer KNN y evaluar con kfold estratificado
final_results = []
for k in (3,5,7,11,21,27,35,51):#,73,103):
    results = kfold_KNN(X_train, Y_train_values, n_splits=5, k=k, verbose=True)
    final_results.append({'K': k, 'results': results})


with open('./resultados8_knn_kfold.json', 'w') as f:
    json.dump(
            {
                'droped_columns': droped_columns,
                'experimentos': final_results
            },
            f, indent=4)


#   Generar un modelo y predecir
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

print('Generando modelo definitivo con k={}'.format(
                                            knn.get_params()['n_neighbors']))
t_start = time.time()

knn.fit(X_train, Y_train_values)
prediction = knn.predict(X_test)

t_stop = time.time()
print('Predición realizada en {} segundos'.format(t_stop-t_start))

Y_test = pd.DataFrame({'id': index_test, 'status_group': prediction})
Y_test.to_csv('prediction7.csv', index=False, header=True, sep=',')
