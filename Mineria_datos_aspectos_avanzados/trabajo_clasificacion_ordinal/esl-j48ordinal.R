library(RWeka)
library(partykit)
library(abind)
library(ggplot2)

set.seed(2)

# Cargar dataset esl
esl <- read.arff('esl.arff')

ordinal.label.encoder <- function(X, class.col, labels.ordered=NULL) {
  
  # Check parameters input
  if (!(is.data.frame(X)) || is.null(colnames(X))) {
    stop('X must be a dataframe with named columns')
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
    stop('Any label in "labels.ordered" not a class of X')
  }
  
  # Nuevo conjunto de etiquetas de clase con información ordinal
  target <- matrix(data=rep(NA, nrow(X)*(length(labels.ordered)-1)),
                    nrow=nrow(X))
  
  # Para todas las etiquetas de la clase
  for (i in 1:(ncol(target))) {
    
    # Los índices para la etiqueta de clase i 
    index.class <- X[,class.column.num]==labels.ordered[i]
    
    #Copiar las etiquetas de la anterior columna ordinal que representa
    # una clase de orden inferior
    if (i>1) {
      target[target[,i-1]==0,i] <- 0
    }
    
    # Generar columna con 0's para la etiqueta i y 1's para el resto
    target[index.class,i] <- 0
    target[is.na(target[,i]),i] <- 1
  }
  
  # Convertir target en columnas de dataframe y agregarlo al df original
  target <- as.data.frame(target)
  target[] <- lapply(target, factor)
  colnames(target) <- paste(colnames(X)[class.column.num],'_', 1:ncol(target), sep='')
  
  # Reemplazar la columna de clase original por las ordinales
  X_new <- cbind(X[,-class.column.num], target)
  
  return(X_new)
}

# Convertir la clase en clases ordinales
esl.ordinal <- ordinal.label.encoder(esl, 'out1')

# De este modo, la columna de clase ha sido convertida en (nº clases-1 ) 8
# columnas de clases ordinales: out1_1,....,out1_8
colnames(esl.ordinal)

# División en conjunto de train y test
train=sample(1:nrow(esl.ordinal), nrow(esl.ordinal)-100)
esl.test=esl.ordinal[-train ,]

# Desarrollar modelo para cada clase ordinal
#### Modelo m1 para out1_1
m1 = J48(out1_1~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m1

# Puesto que en el conjunto de train, sólo 2 instancias pertenecen
# a la clase 1, la salida ordinal out1_1, sólo presenta 2 instancias
# con valor 0, mientras el resto de instancias adopta el valor 1.
# Como consecuencia, se ha originado un clasificador trivial que
# que clasifica todas las instancias como pertenecientes a la clase 1
# de la salida out1_1, es decir, cualquier instancia es clasificada
# como perteneciente a cualquier clase distinta de 1.

#  No se realiza una evaluación del modelo mediante validación cruzada
#  al existir únicamente 2 patrones pertenecientes a la clase 1.

#   El conjunto de test no presentaba ninguna instancia perteneciente a la clase 1.
#   Este clasificador trivial, ha clasificado todas las instancias como diferentes
#   a la clase 1, por lo que el porcentaje de aciertos es del 100%

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m1 <- predict(m1, newdata = esl.test, 'probability')
pred_m1


#### Modelo m2 para out1_2
m2 = J48(out1_2~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m2

# Dibujar el árbol obtenido
plot(m2)

#  Evaluar el modelo sobre el conjunto de test
eval_m2 <- evaluate_Weka_classifier(m2, numFolds = 5, class = TRUE)
eval_m2

# Si bien el modelo generado presenta un rendimiento alto, hay que tener en cuenta que 

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m2 <- predict(m2, newdata = esl.test, 'probability')
pred_m2

#### Modelo m3 para out1_3
m3 = J48(out1_3~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m3
plot(m3)

#  Evaluar el modelo sobre el conjunto de test
eval_m3 <- evaluate_Weka_classifier(m3, numFolds = 5, class = TRUE)
eval_m3

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m3 <- predict(m3, newdata = esl.test, 'probability')
pred_m3

#### Modelo m4 para out1_4
m4 = J48(out1_4~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m4
plot(m4)

#  Evaluar el modelo sobre el conjunto de test
eval_m4 <- evaluate_Weka_classifier(m4, numFolds = 5, class = TRUE)
eval_m4

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m4 <- predict(m4, newdata = esl.test, 'probability')
pred_m4

#### Modelo m5 para out1_5
m5 = J48(out1_5~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m5
plot(m5)

#  Evaluar el modelo sobre el conjunto de test
eval_m5 <- evaluate_Weka_classifier(m5, numFolds = 5, class = TRUE)
eval_m5

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m5 <- predict(m5, newdata = esl.test, 'probability')
pred_m5

#### Modelo m6 para out1_6
m6 = J48(out1_6~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m6
plot(m6)

#  Evaluar el modelo sobre el conjunto de test
eval_m6 <- evaluate_Weka_classifier(m6, numFolds = 5, class = TRUE)
eval_m6

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m6 <- predict(m6, newdata = esl.test, 'probability')
pred_m6

#### Modelo m7 para out1_7
m7 = J48(out1_7~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m7
plot(m7)

#  Evaluar el modelo sobre el conjunto de test
eval_m7 <- evaluate_Weka_classifier(m7, numFolds = 5, class = TRUE)
eval_m7

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m7 <- predict(m7, newdata = esl.test, 'probability')
pred_m7

# Modelo m8 para out1_8
m8 = J48(out1_8~in1+in2+in3+in4,data = esl.ordinal,subset = train)
m8

# Al igual que con el clasificador m1, este modelo la presencia de pocas
# instancias pertencientes a la clase 9, lleva a la construcción de
# un clasificador trivial que clasifica todos los patrones como
# no pertenecientes a la clase 9, es decir, todos los patrones son clasificados
# en la clase 0 de la columna out1_8

#  Evaluar el modelo sobre el conjunto de test
eval_m8 <- evaluate_Weka_classifier(m8, numFolds = 5, class = TRUE)
eval_m8

# El clasificador ha clasificado erróneamente todos los patrones de la
# clase 9

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m8 <- predict(m8, newdata = esl.test, 'probability')
pred_m8

# Una vez realizadas todas las preciciones por parte de todos los clasificadores
# se combinan todas las probabilidades haciendo uso de la probabilidad condicional
# para conocer las probabilidades asignadas conjuntamente por los clasificadores a
# todas las clases.
compute.probability.from.ordinal <- function(P) {
  
  # Función que calcula la probabilidad de clases
  # a partir de un conjunto de probabilidades ordinales
  # dadas por un conjunto de clasificadores binarios
  # ordinales.
  #
  # Recibe: P: Array de (n_instancias x 2 x prob ordinales)
  
  # Calcular número de clases y el de instancias
  n_class <- dim(P)[3] + 1
  n_inst <- dim(P)[1]
  
  # Matriz final con la probabilidades de cada instancia
  # para cada clase
  prob_class <- matrix(data = 1, ncol = n_class, nrow = n_inst)
  
  # Matriz auxiliar para calcular probabilidades condicionales
  aux_cond_prob <- matrix(data = 1, ncol = 1, nrow = n_inst)
  
  # La probabilidad de la clase 1 ya se conoce
  prob_class[,1] <- P[,1,1]
  
  for(i in 2:(n_class-1)) {

    # Añadir y calcular P(C>Ci-1)
    aux_cond_prob = aux_cond_prob*P[,2,i-1]
    
    # Calcular P(C=Ci)
    prob_class[,i] <- aux_cond_prob*P[,1,i]
  }
  
  # Asignar la probabilidad de la úlima clase
  prob_class[,ncol(prob_class)] <- P[,2,n_class-1]*aux_cond_prob
  
  return(prob_class)
}

# Combinar todas las predicciones probabilísticas en un array
pred_all <- abind(pred_m1, pred_m2, pred_m3, pred_m4,
                  pred_m5, pred_m6, pred_m7, pred_m8,
                  along = 3)

# Calcular las probabilidades reales de cada clase mediante
# probabilidad condicional
prob <- compute.probability.from.ordinal(pred_all)

# Determinar las clases a las que han sido clasificados los patrones
pred_class <- max.col(prob)

# Calcular el score en test
score <- mean(pred_class==esl[-train,]$out1)
score

# Conjuntamente, la predicción realizada alcanza un valor de accuraccy
# de 0.62 sobre el conjunto de test

# Analizamos la presencia de cada clase real en el conjunto de test
# y la comparamos con la presencia de cada clase predicha
# con ayuda de un diagrama de barras
score_comparison <- data.frame(value=factor(esl[-train,]$out1),
                               type='real')
score_comparison <- rbind(score_comparison,
                          data.frame(value=factor(pred_class),
                                     type='predicted'))

ggplot2::ggplot(score_comparison, aes(value, fill = type)) +
  ggplot2::geom_bar(position = position_dodge(), colour = 'black') +
  xlab('Clase') +
  ylab('Número de patrones') +
  scale_y_continuous(breaks = scales::pretty_breaks(n=15))

# Se observa que el modelo conjunto generado, carece de capacidad predictiva
# frente a la clase 1 y frente a la clase 9, lo que se explica por la presencia
# de pocos patrones pertenecientes a estas clases en el dataset.

# Comparamos el rendimiento de este modelo con un clasificador J48 no ordinal
esl$out1 <- as.factor(esl$out1)
m.no.ord <- J48(out1~in1+in2+in3+in4, data = esl,subset = train)
m.no.ord

#  Evaluar el modelo sobre el conjunto de test
eval_m.no.ord <- evaluate_Weka_classifier(m.no.ord, numFolds = 5, class = TRUE)
eval_m.no.ord

# Predecir todo el conjunto de test, tomando las probabilidades de pertenencia
pred_m.no.ord <- predict(m.no.ord, newdata = esl[-train,])
score.no.ord <- mean(pred_m.no.ord==esl[-train,]$out1)
score.no.ord

# Se observa que el rendimiento obtenido con este modelo no ordinal,
# es similar al del modelo ordinal. Nuevamente, deseamos comparar las
# predicciones realizadas con la proporción real de clases
score_comparison.no.ord <- data.frame(value=factor(esl[-train,]$out1),
                               type='real')
score_comparison.no.ord <- rbind(score_comparison.no.ord,
                          data.frame(value=factor(pred_m.no.ord),
                                     type='predicted'))

ggplot2::ggplot(score_comparison.no.ord, aes(value, fill = type)) +
  ggplot2::geom_bar(position = position_dodge(), colour = 'black') +
  xlab('Clase') +
  ylab('Número de patrones') +
  scale_y_continuous(breaks = scales::pretty_breaks(n=15))

# Al igual que sucede con el modelo ordinal, no se clasifica ningún
# patrón en la clase 1 debido a la carencia de patrones de esta clase