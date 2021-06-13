library(RWeka)
library(e1071)
library(ggplot2)

# Cargar funciones desarrolladas en el script anterior
source('ordinal.classes.encoder.R')
source('compute.probability.from.ordinal.R')

# Primeramente se elabora la función train.svm.on.ordinal que permite desarrollar
# clasificadores binarios ordinales basados en SVM sobre un dataset,
# admitiendo la misma parametrización que el método svm del paquete e1071.
#
# Por su parte, también se elabora la función predict.svm.on.ordinal que permite usar los
# modelos binarios elaborados y devueltos por train.svm.on.ordinal para predecir
# un conjunto de datos de test

train.svm.on.ordinal <- function(formula, data, subset = NULL, labels.ordered = NULL, ...) {

  # Función que permite entrenar un conjunto
  # de modelos SVM para un dataset con una variable
  # de clase ordinal.
  #
  # El método lanza la implementación de svm del paquete e1071,
  # admitiendo la misma parametrización que este.
  #
  # Recibe:
  # - formula: Relación entre las variables que se pretende desarrollar.
  # - data: Dataframe que contienen los datos a usar para el entrenamiento
  # - subset: Mascara o conjunto de índice que refleja el conjunto de patrones
  #           contenidos en data que, realmente van a ser empleados en entrenamiento.
  #           Dejar a NULL si se desea entrenar con todos los patrones de data
  # - labels.ordered: Valores de la clase expresados
  #               según el orden en el que deben de
  #               ordenados de izquierda a derecha.
  #               Si no se especifica, se tratará
  #               de obtener de la propia columna.
  # Resto de parámetros admitidos por la implementación de svm de e1071
  # Devuelve:
  # Lista con los campos "models" y "parameters"
  #   - parameters: Lista de parámetros con los que se ha elaborado los modelos
  #   - models: Lista con los modelos binarios ordinales elaborados
  
  # Tomar variables independiente e independientes< de la formula
  v.dep <- as.character(formula[2])
  v.ind.sel <- as.character(formula[-2][2]) #attr(terms(formula), 'term.labels')
  v.ind <- colnames(data)[which(colnames(data)!=v.dep)]
  
  if (!v.dep %in% colnames(data)) {
    stop(paste0(v.dep,' not a columns of data'))
  }
  
  # Codificar el dataset en clases binarias ordinales
  data_ord <- ordinal.classes.encoder(data, v.dep, labels.ordered)
  
  # Tomar los valores de la clase si no se han proporcionado
  if (is.null(labels.ordered)) {
    labels.ordered <- sort(unique(data[[v.dep]]))
  }
  
  
  # Tomar los nombres de las columnas de clase nominales
  class_ord <- grep(pattern = paste0(v.dep,'_[0-9]+'), colnames(data_ord))
  class_ord <- colnames(data_ord)[class_ord]
  
  # Generar clasificadores binarios para cada clase ordinal binaria
  ord_models <- list(parameters=list(),
                     models=list())
  
  for (cl in class_ord) {
    
    # Determinar la fórmula de este clasificador binario
    form <- as.formula(paste0(cl,'~',v.ind.sel))
    
    m <- tryCatch({
          do.call(e1071::svm, list(formula = form,
                                   data = data_ord[,which(colnames(data_ord) %in%
                                                          c(v.ind,cl))],
                                   subset = subset,
                                   probability = TRUE,
                                   ...))
    }, error = function(cond) {

      if ('Model is empty!' %in% cond) {
        message(cond) # Mostrar mensaje de error

        # No se puede elaborar un clasificador binario y
        # se clasifica a la clase mayoritaria
        m <- as.integer(names(sort(table(data_ord[subset,][cl]),
                                   decreasing = T)[1]))
        class(m) <- 'trivial_classifier'
        return(m)
      }
      else {
        stop(cond)
      }
    })

    ord_models$models[[paste0('m_',cl)]] <- m
  }
  
  # Registrar el nombre de la columna de clase original en la lista
  ord_models$parameters[['class.name']] <- v.dep
  ord_models$parameters[['labels.ordered']] <- labels.ordered
  
  return(ord_models)
}

predict.svm.on.ordinal <- function(models, data) {
  
  # Función para predecir un conjunto de datos para los que
  # es asignable una clase ordinal
  
  v.dep <- models$parameters[['class.name']]
  labels.ordered <- models$parameters$labels.ordered
  
  # Codificar el conjunto de datos en clases binarias ordinales
  data_ord <- data#ordinal.classes.encoder(data, v.dep, labels.ordered)
  
  # Para almacenar las prediciones
  ord_pred <- array(data=NA,
                  dim = c(nrow(data_ord),2,length(models$models)))
  dimnames(ord_pred)[[1]] <- row.names(data_ord)
  
  # Predecir el conjunto de datos por cada clasificador ordinal binario
  for (i in 1:length(models$models)) {
    if (any(class(models$models[[i]]) == 'trivial_classifier')) {
      # Un clasificador trivial
      index.class <- models$models[[i]] + 1
      ord_pred[,index.class,i] <- 1
      ord_pred[,-index.class,i] <- 0
    } else {
      # Un clasificador SVM
      pred <- attr(predict(models$models[[i]], data_ord, probability = TRUE),
                   'probabilities')

      ord_pred[,2,i] <- pred[,which(colnames(pred)=='1')] #1
      ord_pred[,1,i] <- pred[,which(colnames(pred)=='0')] #0
    }
  }

  # Calcular probabilidades reales de cada clase
  prob_pred <- compute.probability.from.ordinal(ord_pred)
  
  # Calcular las clases predichas
  return(labels.ordered[max.col(prob_pred)])
}

#######################

### Ahora se prueban las anteriores funciones con varios datasets
set.seed(11)

#### Dataset esl
# Cargar dataset esl
esl <- read.arff('esl.arff')

# División en conjunto de train y test
train.esl=sample(1:nrow(esl), nrow(esl)-100)
esl.test=esl[-train.esl,]

# Entrenamiento con parámetros por defecto
models.esl <- train.svm.on.ordinal(out1~., esl, subset = train.esl)

# Predicción y evaluación del conjunto de test
pred.esl <- predict.svm.on.ordinal(models.esl, esl.test)

# Cálculo del accuracy
acc.esl <- mean(pred.esl == esl.test$out1)
cat('Accuraccy del modelo obtenido para esl: ',acc.esl)

# El rendimiento del modelo es alto (74% de accuracy),

# Evaluamos ahora el rendimiento de otro modelo no ordinal
esl$out1 <- factor(esl$out1)
model.no.ord.esl <- svm(out1~., esl, subset = train.esl)

#   Cálculo del accuracy en test
pred.no.ord.esl <- predict(model.no.ord.esl, esl.test)
acc.no.ord.esl <- mean(pred.no.ord.esl == esl.test$out1)
cat('Accuraccy del modelo obtenido para esl: ',acc.no.ord.esl)

# Se ha obtenido un modelo de rendimiento superior

# Se prueba a comparar el procentaje de prediciones del modelo ordinal y
# el modelo no ordinal sobre el dataset esl, comparando esta proporciones
# a su vez, con las proporciones reales de las clases
score_comp.no.ord.esl <- data.frame(value=factor(esl[-train,]$out1),
                                      type='real')
score_comp.no.ord.esl <- rbind(score_comp.no.ord.esl,
                                 data.frame(value=factor(pred.esl),
                                            type='predicted ordinal model'))
score_comp.no.ord.esl <- rbind(score_comp.no.ord.esl,
                               data.frame(value=factor(pred.no.ord.esl),
                                          type='predicted no-ordinal model'))

ggplot2::ggplot(score_comp.no.ord.esl, aes(value, fill = type)) +
  ggplot2::geom_bar(position = position_dodge(), colour = 'black') +
  xlab('Clase') +
  ylab('Número de patrones') +
  scale_y_continuous(breaks = scales::pretty_breaks(n=15))

# Ninguno de los dos modelos desarrollados, clasifica patrones en la clase
# 1 y en la clase 9 debido a la carencia de patrones pertenecientes a estas
# clases en los conjuntos de train
#
# El modelo no ordinal supera en rendimiento al modelo ordinal: La inclusión
# de información de orden entre las clases, perjudica el rendimiento del
# clasificador

#### Dataset era
# Cargar dataset era
era <- read.arff('era.arff')

# Analizamos el dataset era
str(era)

# La clase en este dataset queda especificada también en la columna
# out1, habiendo un total de 9 clases
sort(unique(era$out1))

# División en conjunto de train y test
train.era=sample(1:nrow(era), nrow(era)-100)
era.test=era[-train.era,]

# Entrenamiento con parámetros por defecto
models.era <- train.svm.on.ordinal(out1~., era, subset = train.era)

# Predicción y evaluación del conjunto de test
pred.era <- predict.svm.on.ordinal(models.era, era.test)

# Cálculo del accuracy
acc.era <- mean(pred.era == era.test$out1)
cat('Accuraccy del modelo obtenido para era: ',acc.era)

# El modelo obtenido presenta un rendimiento muy bajo, nuevamente,
# se compara el rendimiento de este modelo con el de un modelo
# no ordinal
era$out1 <- factor(era$out1)
model.no.ord.era <- svm(out1~., era, subset = train.era)

#   Cálculo del accuracy en test
pred.no.ord.era <- predict(model.no.ord.era, era.test)
acc.no.ord.era <- mean(pred.no.ord.era == era.test$out1)
cat('Accuraccy del modelo obtenido para esl: ',acc.no.ord.era)

# El rendimiento del modelo obtenido es reducido (14%)

# Se prueba a comparar el procentaje de prediciones del modelo ordinal y
# el modelo no ordinal sobre el dataset esl, comparando esta proporciones
# a su vez, con las proporciones reales de las clases
score_comp.no.ord.era <- data.frame(value=factor(era[-train.era,]$out1),
                                    type='real')
score_comp.no.ord.era <- rbind(score_comp.no.ord.era,
                               data.frame(value=factor(pred.era),
                                          type='predicted ordinal model'))
score_comp.no.ord.era <- rbind(score_comp.no.ord.era,
                               data.frame(value=factor(pred.no.ord.era),
                                          type='predicted no-ordinal model'))

ggplot2::ggplot(score_comp.no.ord.era, aes(value, fill = type)) +
  ggplot2::geom_bar(position = position_dodge(), colour = 'black') +
  xlab('Clase') +
  ylab('Número de patrones') +
  scale_y_continuous(breaks = scales::pretty_breaks(n=15))

# Se observa que el modelo ordinal presenta problemas para
# clasificar patrones de la clase 8 y 9

#### Dataset lev
# Cargar dataset lev
lev <- read.arff('lev.arff')

# Analizamos el dataset lev
str(lev)

# La clase en este dataset queda especificada en la columna Out1
# se distinguen un total de 5 clases: 0, 1, 2, 3 y 4
sort(unique(lev$Out1))

# División en conjunto de train y test
train.lev=sample(1:nrow(lev), nrow(lev)-100)
lev.test=lev[-train.lev,]

# Entrenamiento con parámetros por defecto
models.lev <- train.svm.on.ordinal(Out1~., lev, subset = train.lev)

# Predicción y evaluación del conjunto de test
pred.lev <- predict.svm.on.ordinal(models.lev, lev.test)

# Cálculo del accuracy
acc.lev <- mean(pred.lev == lev.test$Out1)
cat('Accuraccy del modelo obtenido para lev: ',acc.lev)

#### Dataset swd
# Cargar dataset lev
swd <- read.arff('swd.arff')

# Analizamos el dataset swd
str(swd)

# La clase en este dataset queda especificada en la columna Out1
# se distinguen un total de 4 clases: 2, 3, 4 y 5
sort(unique(swd$Out1))

# División en conjunto de train y test
train.swd=sample(1:nrow(swd), nrow(swd)-100)
swd.test=swd[-train.swd,]

# Entrenamiento con parámetros por defecto
models.swd <- train.svm.on.ordinal(Out1~., swd, subset = train.swd)

# Predicción y evaluación del conjunto de test
pred.swd <- predict.svm.on.ordinal(models.swd, swd.test)

# Cálculo del accuracy
acc.swd <- mean(pred.swd == swd.test$Out1)
cat('Accuraccy del modelo obtenido para swd: ',acc.swd)