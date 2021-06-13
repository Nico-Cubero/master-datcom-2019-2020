# Librerías importadas
library('caret')
library('class')
library('MASS')
library('dplyr')
library('MVTests')

# Función para leer el dataset vehicle
read.vehicle.dataset <- function(filename) {
  # Cargar datos
  dat <- read.table(filename, comment.char="@", sep=',')
  
  # Asignar nombres
  names(dat) <- c('Compactness', 'Circularity', 'Distance_circularity',
                  'Radius_ratio', 'Praxis_aspect_ratio',
                  'Max_length_aspect_ratio', 'Scatter_ratio',
                  'Elongatedness', 'Praxis_rectangular', 'Length_rectangular',
                  'Major_variance', 'Minor_variance', 'Gyration_radius',
                  'Major_skewness', 'Minor_skewness', 'Minor_kurtosis',
                  'Major_kurtosis', 'Hollows_ratio', 'Class')
  
  return(dat)
}

# Cargar el dataset vehicle
vehicle <- read.vehicle.dataset('./Datasets Clasificacion/vehicle/vehicle.dat')
fold_basename <- './Datasets Clasificacion/vehicle/vehicle'

# Modelos basados en k-NN

# Evaluar los modelos mediante validación cruzada 10-fold
run_knn_k_fold_vehicle <- function(i, x, k, standarizate=NULL, tt = "test") {
  # Cargar conjuntos de entrenamiento
  file <- paste(x, "-10-", i, "tra.dat", sep=""); 
  x_tra <- read.csv(file, comment.char="@")
  
  # Cargar conjuntos de test
  file <- paste(x, "-10-", i, "tst.dat", sep="");
  x_tst <- read.csv(file, comment.char="@")
  
  names(x_tra) <- c('Compactness', 'Circularity', 'Distance_circularity',
                    'Radius_ratio', 'Praxis_aspect_ratio',
                    'Max_length_aspect_ratio', 'Scatter_ratio',
                    'Elongatedness', 'Praxis_rectangular', 'Length_rectangular',
                    'Major_variance', 'Minor_variance', 'Gyration_radius',
                    'Major_skewness', 'Minor_skewness', 'Minor_kurtosis',
                    'Major_kurtosis', 'Hollows_ratio', 'Class')
  
  names(x_tst) <- names(x_tra)
  
  # Estandarizar las variables proporcionadas
  for (i in standarizate) {
    x_tra[i] <- scale(x_tra[i])
    x_tst[i] <- scale(x_tst[i])
  }
  
  print(summary(x_tra))
  
  if (tt == "train") { test <- x_tra }
  else { test <- x_tst }
  
  # Entrenar el modelo sobre el conjunto de train
  model.eval <- knn(train=x_tra[,-ncol(x_tra)], test=x_tst[,-ncol(x_tst)],
                    cl=x_tra[,ncol(x_tra)], k=k)
  
  # Evaluación del CCR sobre test
  postResample(model.eval, test$Class)[1]
}

# Modelo con k=7
knn.model.fit1 <- knn(train=vehicle[,-ncol(vehicle)], test=vehicle[,-ncol(vehicle)],
                      cl=vehicle[,ncol(vehicle)], k=7)

# Evaluar error CCR train
cat('Accuracy sobre el conjunto de train: ',postResample(knn.model.fit1,
                                                      vehicle$Class)[1],
    fill=T)
cat('Accuracy sobre 10-fold: ',mean(sapply(1:10,run_knn_k_fold_vehicle,
                                fold_basename, 7)), fill=T)



# Modelo con k=7, con atributos escalados
vehicle.rescaled <- vehicle
vehicle.rescaled[,-ncol(vehicle)] <- scale(vehicle.rescaled[,-ncol(vehicle)])

knn.model.fit2 <- knn(train=vehicle.rescaled[,-ncol(vehicle.rescaled)],
                      test=vehicle.rescaled[,-ncol(vehicle.rescaled)],
                      cl=vehicle.rescaled[,ncol(vehicle.rescaled)], k=7)

# Evaluar error CCR train
cat('Accuracy sobre el conjunto de train: ',postResample(knn.model.fit2,
                                            vehicle.rescaled$Class)[1],
    fill=T)
cat('Accuracy sobre 10-fold: ',mean(sapply(1:10,run_knn_k_fold_vehicle,
                                           fold_basename, 7,
                                           colnames(vehicle)[1:ncol(vehicle)-1])),
                                fill=T)



# Valores de K usados en las pruebas
k.values <- c(1,3,7,13,21,51,75,103)

evaluate.best_k.knn <- function(k) {
  
  # Evaluar el modelo
  #model.eval <-  knn(train=vehicle.rescaled[,-ncol(vehicle.rescaled)],
   #                  test=vehicle.rescaled[,-ncol(vehicle.rescaled)],
    #                 cl=vehicle.rescaled[,ncol(vehicle.rescaled)], k)

  model.eval <-  knn(train=vehicle[,-ncol(vehicle)],
                     test=vehicle[,-ncol(vehicle)],
                     cl=vehicle[,ncol(vehicle)], k)

  meausures <- c(k,                                              # Valor de k
                 postResample(model.eval, vehicle$Class)[1],     # CCR train
                 mean(sapply(1:5,run_knn_k_fold_vehicle,           # CCR 10-fold
                             fold_basename, k)))
                             #colnames(vehicle)[1:ncol(vehicle)-1])))
  names(meausures) <- c('k', 'CCR', 'CCR 10-fold')
  
  return(meausures)
}

# Aplicar batería de pruebas para cada valor de k. En este caso no interesa
# usar el parámetro tuneGrid ya que no nos permitiría calcular las métricas
# sobre todo el conjunto del dataset
results <- sapply(k.values, FUN=evaluate.best_k.knn)
results

# Modelos LDA

# Analizar las varianzas por clase
vehicle.bus <- vehicle %>% filter(Class==' bus ')
vehicle.opel <- vehicle %>% filter(Class==' opel')
vehicle.saab <- vehicle %>% filter(Class==' saab')
vehicle.van <- vehicle %>% filter(Class==' van ')

var.class <- rbind(apply(vehicle.bus[,-ncol(vehicle.bus)], FUN=var, MARGIN = 2),
                   apply(vehicle.opel[,-ncol(vehicle.opel)], FUN=var, MARGIN = 2),
                   apply(vehicle.saab[,-ncol(vehicle.saab)], FUN=var, MARGIN = 2),
                   apply(vehicle.van[,-ncol(vehicle.van)], FUN=var, MARGIN = 2))

rownames(var.class) <- c('bus', 'opel', 'saab', 'van')
var.class

# Analizar las equidad en las covarianzas con el test de BoxM
test.boxm <- BoxM(vehicle[,-ncol(vehicle)], group=vehicle$Class)
cat('p-value asociado al test BoxM: ',test.boxm$p.value)

# Función para validación cruzada
run_lda_k_fold_vehicle <- function(i, x, model, tt = "test") {
  # Cargar conjuntos de entrenamiento
  file <- paste(x, "-10-", i, "tra.dat", sep=""); 
  x_tra <- read.vehicle.dataset(file)
  
  # Cargar conjuntos de test
  file <- paste(x, "-10-", i, "tst.dat", sep="");
  x_tst <- read.vehicle.dataset(file)
  
  if (tt == "train") { test <- x_tra }
  else { test <- x_tst }
  
  # Entrenar el modelo sobre el conjunto de train
  form <- terms(model)
  model.eval <- lda(formula=form, data=x_tra)
  
  # Evaluación del CCR sobre test
  pred <- predict(model.eval, test)
  postResample(pred$class, test$Class)[1]
}

# Class ~ .
lda.model.fit1 <- lda(Class~.-Scatter_ratio-Praxis_rectangular-Length_rectangular-
                        Minor_variance-Gyration_radius, data=vehicle)
lda.model.fit1

lda.model.fit1.pred <- predict(lda.model.fit1, vehicle)
cat('CCR medido sobre entrenamiento:', postResample(lda.model.fit1.pred$class,
                                                    vehicle$Class)[1], fill=T)
cat('Accuracy sobre 10-fold: ',mean(sapply(1:10,run_lda_k_fold_vehicle,
                                           fold_basename, lda.model.fit1)),fill=T)
# Class ~ .
lda.model.fit2 <- lda(Class~., data=vehicle)
lda.model.fit2

lda.model.fit2.pred <- predict(lda.model.fit2, vehicle)
cat('CCR medido sobre entrenamiento:', postResample(lda.model.fit2.pred$class,
                                                    vehicle$Class)[1], fill=T)
cat('Accuracy sobre 10-fold: ',mean(sapply(1:10,run_lda_k_fold_vehicle,
                                           fold_basename, lda.model.fit2)),fill=T)

# Modelos QDA

# Analizar las correlaciones entre las variables para cada clase
corr.bus <- cor(vehicle.bus[,-ncol(vehicle.bus)], method = 'kendall')
corr.opel <- cor(vehicle.opel[,-ncol(vehicle.opel)], method = 'kendall')
corr.saab <- cor(vehicle.saab[,-ncol(vehicle.saab)], method = 'kendall')
corr.van <- cor(vehicle.van[,-ncol(vehicle.van)], method = 'kendall')

pdf('Correlograma_kendall_vehicle_clases.pdf', width=14, height = 15)
par(mfrow=c(2,2))
corrplot(corr.bus, method='color', type='lower', number.cex = 0.8,
         title='bus', addCoef.col='gray', mar=c(0,0,1,0))
corrplot(corr.opel, method='color', type='lower', number.cex = 0.80,
         title='opel', addCoef.col='gray', mar=c(0,0,1,0))
corrplot(corr.saab, method='color', type='lower', number.cex = 0.80,
         title='saab', addCoef.col='gray', mar=c(0,0,1,0))
corrplot(corr.van, method='color', type='lower', number.cex = 0.80,
         title='van',addCoef.col='gray', mar=c(0,0,1,0))
dev.off()

# Elaborar una función de validación para qda
run_qda_k_fold_vehicle <- function(i, x, model, tt = "test") {
  # Cargar conjuntos de entrenamiento
  file <- paste(x, "-10-", i, "tra.dat", sep=""); 
  x_tra <- read.vehicle.dataset(file)
  
  # Cargar conjuntos de test
  file <- paste(x, "-10-", i, "tst.dat", sep="");
  x_tst <- read.vehicle.dataset(file)
  
  if (tt == "train") { test <- x_tra }
  else { test <- x_tst }
  
  # Entrenar el modelo sobre el conjunto de train
  form <- terms(model)
  model.eval <- qda(formula=form, data=x_tra)
  
  # Evaluación del CCR sobre test
  pred <- predict(model.eval, test)
  postResample(pred$class, test$Class)[1]
}

# Elaborar un modelo basado en QDA
# Class ~ .
qda.model.fit1 <- qda(Class~., data=vehicle)
qda.model.fit1

qda.model.fit1.pred <- predict(qda.model.fit1, vehicle)
cat('CCR medido sobre entrenamiento:', postResample(qda.model.fit1.pred$class,
                                                    vehicle$Class)[1], fill=T)
cat('Accuracy sobre 10-fold: ',mean(sapply(1:10,run_qda_k_fold_vehicle,
                                           fold_basename, qda.model.fit1)),fill=T)
