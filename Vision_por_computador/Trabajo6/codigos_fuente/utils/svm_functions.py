#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:47:06 2020

@author: nico
"""

# Importación de librerías
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, roc_auc_score)
from cv2.ml import SVM_create, ROW_SAMPLE

def train_svm(X, y, svm_params=None):

    """
        Entrena el clasificador

        Parameters:
        training_data (np.array): datos de entrenamiento
        classes (np.array): clases asociadas a los datos de entrenamiento

        Returns:
        cv2.SVM: un clasificador SVM
    """

    svm = SVM_create()

    if isinstance(svm_params, dict):
        if 'kernel_type' in svm_params:
            svm.setKernel(svm_params['kernel_type'])

        if 'svm_type' in svm_params:
            svm.setType(svm_params['svm_type'])

        if 'Cvalue' in svm_params:
            svm.setC(svm_params['Cvalue'])

        if 'degree' in svm_params:
            svm.setDegree(svm_params['degree'])

        if 'gamma' in svm_params:
            svm.setGamma(svm_params['gamma'])

        if 'coef0' in svm_params:
            svm.setCoef0(svm_params['coef0'])

    svm.train(X, ROW_SAMPLE, y)

    return svm

def test_svm(model, X_test, y_test):

    meausures = {}

    # Predecir todos los patrones de test
    y_pred = model.predict(X_test)[1]

    # Calcular medidas de bondad
    meausures['CCR'] = accuracy_score(y_test, y_pred)#np.mean(y_pred==y_test)
    meausures['precission'] = precision_score(y_test, y_pred)
    meausures['recall'] = recall_score(y_test, y_pred)
    meausures['ROC'] = roc_auc_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)

    meausures['confussion_matrix'] = {
                                        'TN': float(conf_matrix[0,0]),
                                        'FN': float(conf_matrix[0,1]),
                                        'FP': float(conf_matrix[1,0]),
                                        'TP': float(conf_matrix[1,1])
                                      }

    return meausures
