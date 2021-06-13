# -*- coding: utf-8 -*-
"""
	@author Nicolás Cubero
"""

# Importación de librerías
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
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

#   Añadir date_recorded separado por año, días y fila
X_train[['date_recorded_year',
         'date_recorded_month',
         'date_recorded_day']] = (X_train['date_recorded'].str.split('-',
                                 expand=True)).astype(np.int64)
X_test[['date_recorded_year',
        'date_recorded_month',
        'date_recorded_day']] = (X_test['date_recorded'].str.split('-',
                                expand=True)).astype(np.int64)

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

#   Almacenar los tipos de cada columna
dtypes = dict(zip(X_train.columns, X_train.dtypes))

# Separar variables numéricas y categóricas
X_train_cat = X_train[X_train.columns[(X_train.dtypes == 'object').values]]
X_train_num = X_train[X_train.columns[(X_train.dtypes != 'object').values]]

X_test_cat = X_test[X_test.columns[(X_test.dtypes == 'object').values]]
X_test_num = X_test[X_test.columns[(X_test.dtypes != 'object').values]]

dtypes_cat = dict(zip(X_train_cat.columns, X_train_cat.dtypes))
dtypes_num = dict(zip(X_train_num.columns, X_train_num.dtypes))

# Imputación de valores perdidos
imputer = SimpleImputer(strategy='most_frequent')

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns,
                       index=X_train.index)
X_train = X_train.astype(dtypes)

X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns,
                      index=X_test.index)
X_test = X_test.astype(dtypes)

# Separar variables numéricas y categóricas
X_train_cat = X_train[X_train.columns[(X_train.dtypes == 'object').values]]
X_train_num = X_train[X_train.columns[(X_train.dtypes != 'object').values]]

X_test_cat = X_test[X_test.columns[(X_test.dtypes == 'object').values]]
X_test_num = X_test[X_test.columns[(X_test.dtypes != 'object').values]]

#   Normalizar
normalizer = Normalizer()
normalizer.fit(X_train_num)

X_train_num = normalizer.transform(X_train_num)
X_test_num = normalizer.transform(X_test_num)

# Binarizar categóricas
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train_cat)

X_train_cat = encoder.transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

X_train = hstack((X_train_cat, X_train_num)).tocsr()
X_test = hstack((X_test_cat, X_test_num)).tocsr()

# Aplicar reducción de dimensionalidad con PCA en una batería de pruebas
final_results = []
comp_cand = (X_train.shape[1]-5000, X_train.shape[1]-10000,
             X_train.shape[1]-50000)
for comp in comp_cand:
    
    #   Añadir una entrada para almacenar los resultados para el número
    #   de componentes
    final_results.append({
                            'PCA': comp,
                            'resultados': []
                        })
    
    #   Ejecutar TruncatedSVD
    pca = TruncatedSVD(n_components=comp)

    X_train_red = pca.fit_transform(X_train)
    
    # Probar a hacer KNN y evaluar con kfold estratificado

    for k in (3,5,7):#,,21,27,35,5173,103):
        results = kfold_KNN(X_train_red, Y_train_values, n_splits=5, k=k,
                            n_jobs=-1, verbose=True)
        final_results[-1]['resultados'].append({'K': k, 'results': results})
    
    
    with open('./resultados12_knn_kfold.json', 'w') as f:
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
