library('RWeka')
library('xgboost')

#### Implementación de funciones

# Primeramente, se desarrollarán las funciones necesarias para elaborar
# un conjunto de modelos binarios ordinales basados en xgboost con
# restricciones de monotócidad

ordinal.monotonic.classes.encoder <- function(X, class.col, labels.ordered=NULL) {
  
  # Función para la transformación de la variable
  # de clase categórica de un dataset en variables
  # de clase binarias ordinales dada una relación
  # de orden entre los valores de clase indicada
  # y con restricciones de monotocidad consistente
  #
  # Recibe:
  # X: Dataset con una variable de clase categórico
  #     debe de pasarse como dataframe con nombres
  #     de columna no nulos.
  # class.col: Nombre o número de la columna del dataframe
  #           que expresa la clase.
  # labels.orderd: Valores de la clase expresados
  #               según el orden en el que deben de
  #               ordenados de izquierda a derecha.
  #               Si no se especifica, se tratará
  #               de obtener de la propia columna.
  #
  # Devuelve: Dataframe con las variables independientes
  #           del dataset y las variables de clase binarias
  #           ordinales.
  
  # Check parameters input
  if (!(is.data.frame(X)) || is.matrix(X)) {
    stop('X must be a dataframe or a matrix')
  } else if (is.null(colnames(X))) {
    colnames(X) <- paste0('C',1:ncol(X))
  }
  
  if (length(class.col)!=1) {
    stop('Only one value allowed for "label.col"')
  }
  
  # Número de columna de clase
  class.column.num <- NA
  
  # Tomar el número de la columna de clase
  if (is.character(class.col)) {
    
    if (!class.col %in% colnames(X)) {
      stop(paste('"',class.col,'" not included in dataset'))
    }
    
    class.column.num <- which(colnames(X)==class.col)
    
  } else if (is.integer(class.col)) {
    
    if (class.col < 1 || class.col>ncol(X)) {
      stop('Column ',class.col,' not a valid column')
    }
    
    class.column.num <- class.col
    
  } else {
    stop('Only character or integers allowed for "class.col"')
  }
  
  # Las etiquetas pasadas deben de ser valores de etiqueta de la clase
  if (is.null(labels.ordered)) {
    
    # Tomar orden de los valores de la columna de clase si ningún orden es especificado
    if (is.ordered(X[,class.column.num]) || is.numeric(X[,class.column.num])) {
      labels.ordered <- sort(unique(X[,class.column.num]))
    }
    else {
      stop('Cannot deduce implicit order from class')
    }
    
  } else if (any(!labels.ordered %in% unique(X[,class.column.num]))) {
    warning('Any label in "labels.ordered" not a class of X')
  }
  
  # Nuevo conjunto de etiquetas de clase con información ordinal monotónica consistente
  target <- matrix(data=rep(0, nrow(X)*(length(labels.ordered)-1)),
                   nrow=nrow(X),ncol = (length(labels.ordered)-1))
  
  # Para todas las etiquetas de la clase
  for (i in 1:(ncol(target))) {
    
    c <- i+1 #  Valor real de clase
    
    # Los índices para la etiqueta de clase i 
    index.class <- X[,class.column.num]>=labels.ordered[c]
    
    # Generar columna con 1's para los valores mayores o iguales y 0's para el resto
    target[index.class,i] <- 1
  }
  
  # Convertir target en columnas de dataframe y agregarlo al df original
  target <- as.data.frame(target)
  #target[] <- lapply(target, factor, ordered=TRUE)
  
  colnames(target) <- paste(colnames(X)[class.column.num],'_', 1:ncol(target)+1, sep='')
  
  # Reemplazar la columna de clase original por las ordinales
  X_new <- cbind(X[,-class.column.num], target)
  
  return(X_new)
}

train.monotonic.xgboost.on.ordinal <- function(data, label, labels.ordered = NULL, ...) {
  
  # Función que permite entrenar un conjunto
  # de modelos xgboost binario monotónico para
  # un dataset con una variable de clase ordinal.
  #
  # El método lanza la implementación de xgboost del paquete xgboost,
  # admitiendo la misma parametrización que este.
  #
  # Recibe:
  # - data: Matriz que contienen los datos a usar para el entrenamiento
  # - label: Columna con las etiquetas a predecir.
  # - labels.ordered: Valores de la clase expresados
  #               según el orden en el que deben de
  #               ordenados de izquierda a derecha.
  #               Si no se especifica, se tratará
  #               de obtener de la propia columna.
  # Resto de parámetros admitidos por la implementación de xgboost de xgboost
  # Devuelve:
  # Lista con los campos "models" y "parameters"
  #   - parameters: Lista de parámetros con los que se ha elaborado los modelos
  #   - models: Lista con los modelos binarios ordinales elaborados
  
  # Codificar el dataset en clases binarias ordinales
  label_ord <- ordinal.monotonic.classes.encoder(as.data.frame(label), 1L,
                                                 labels.ordered)
  
  
  # Tomar los valores de la clase si no se han proporcionado
  if (is.null(labels.ordered)) {
    labels.ordered <- sort(unique(label))
  }
  
  # Generar clasificadores binarios para cada clase ordinal binaria
  mono_models <- list(parameters=list(),
                     models=list())
  
  for (cl in 1:ncol(label_ord)) {

    # Entrenar modelo xgboost y almacenarlo en la lista
    m <- do.call(xgboost, list(data = as.matrix(data),
                               label = label_ord[,cl],
                               objective = 'binary:logistic',
                               monotone_constraints = 1,
                               ...))

    mono_models$models[[paste0('m_',(cl+1))]] <- m
  }
  
  # Registrar el orden de los valores de clase especificados (en tal caso)
  mono_models$parameters[['labels.ordered']] <- labels.ordered
  
  return(mono_models)
}

predict.monoxgboost.on.ordinal <- function(models, X){
  
  labels.ordered <- models$parameters$labels.ordered
  
  # Para almacenar las predicciones de cada instancia
  mono_pred <- rep(1, nrow(X))
  names(mono_pred) <- row.names(X)
  
  # Usar cada clasificador binario para clasificar los patrones
  for (cl in 1:length(models$models)) {
    pred <- predict(models$models[[cl]], as.matrix(X))
    pred <- ifelse(pred>=0.5,1,0)

    # Combinar la predicción con la predicción global dada la relación:
    # h(c) = 1 + sum h(c)
    mono_pred <- mono_pred + pred
  }
  
  return(labels.ordered[mono_pred])
}

#######################################

### Ahora se prueban las anteriores funciones con distintos datasets
set.seed(11)

# Cargar dataset esl
esl <- read.arff('esl.arff')

# División en conjunto de train y test
train=sample(1:nrow(esl), nrow(esl)-100)
esl.test=esl[-train ,]

# Entrenamiento con parámetros por defecto
models.esl <- train.monotonic.xgboost.on.ordinal(data=esl[train,-ncol(esl)],
                                             label=esl[train,ncol(esl)],
                                             labels.ordered = 1:9,
                                             nrounds=10)

# Predicción y evaluación del conjunto de test
pred.esl <- predict.monoxgboost.on.ordinal(models.esl, esl.test[,-ncol(esl.test)])

# Calculo del accuracy
accuraccy <- mean(pred.esl == esl.test$out1)
accuraccy

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
models.era <- train.monotonic.xgboost.on.ordinal(data=era[train,-ncol(era)],
                                                 label=era[train,ncol(era)],
                                                 labels.ordered = 1:9,
                                                 nrounds=10)

# Predicción y evaluación del conjunto de test
pred.era <- predict.monoxgboost.on.ordinal(models.era, era.test[,-ncol(era.test)])

# Cálculo del accuracy
acc.era <- mean(pred.era == era.test$out1)
cat('Accuraccy del modelo obtenido para era: ',acc.era)


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
models.lev <- train.monotonic.xgboost.on.ordinal(data=lev[train,-ncol(lev)],
                                                 label=lev[train,ncol(lev)],
                                                 #labels.ordered = 1:9,
                                                 nrounds=10)

# Predicción y evaluación del conjunto de test
pred.lev <- predict.monoxgboost.on.ordinal(models.lev, lev.test[,-ncol(lev.test)])

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
models.swd <- train.monotonic.xgboost.on.ordinal(data=swd[train,-ncol(swd)],
                                                 label=swd[train,ncol(swd)],
                                                 nrounds=10)

# Predicción y evaluación del conjunto de test
pred.swd <- predict.monoxgboost.on.ordinal(models.swd, swd.test[,-ncol(swd.test)])

# Cálculo del accuracy
acc.swd <- mean(pred.swd == swd.test$Out1)
cat('Accuraccy del modelo obtenido para swd: ',acc.swd)
