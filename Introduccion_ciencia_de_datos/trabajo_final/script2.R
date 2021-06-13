###############################################################################
# Nombre del script: script2.R
# Desarrollado por: Nicolás Cubero
# Descripción: Script para el desarrollo de modelos de regresión sobre el
# dataset house
# Nota: Este script ha sido desarrollado para ejecutarse de forma interactiva
#       línea por línea
###############################################################################
# Librerías cargadas
library('ggplot2')
library('gridExtra')

library('kknn')

read.house.dataset <- function(filename) {
  # Cargar datos
  dat <- read.table(filename, comment.char="@", sep=',')
  
  # Asignar nombres
  names(dat) <- c('P1', 'P5p1', 'P6p2', 'P11p4', 'P14p9', 'P15p1', 'P15p3',
                  'P16p2', 'P18p2', 'P27p4', 'H2p2', 'H8p2', 'H10p1', 'H13p1',
                  'H18pA', 'H40p4', 'Price')
  
  return(dat)
}

# Cargar el dataset house
house <- read.house.dataset('./Datasets\ Regresion/house/house.dat')

# Definición de la función para evaluar un modelo lineal
evaluate.rmse.house <- function(model) {
  yprime = predict(model, house)
  sqrt(sum(abs(house$Price-yprime)^2)/length(yprime)) #Calcular RMSE
}

# Modelos lineales
lineal.simple.P1 <- lm(Price ~ P1, data=house)
lineal.simple.P5p1 <- lm(Price ~ P5p1, data=house)
lineal.simple.P6p2 <- lm(Price ~ P6p2, data=house)
lineal.simple.P11p4 <- lm(Price ~ P11p4, data=house)
lineal.simple.P14p9 <- lm(Price ~ P14p9, data=house)
lineal.simple.P14p9 <- lm(Price ~ P14p9, data=house)
lineal.simple.P15p1 <- lm(Price ~ P15p1, data=house)
lineal.simple.P15p3 <- lm(Price ~ P15p3, data=house)
lineal.simple.P16p2 <- lm(Price ~ P16p2, data=house)
lineal.simple.P18p2 <- lm(Price ~ P18p2, data=house)
lineal.simple.P27p4 <- lm(Price ~ P27p4, data=house)
lineal.simple.H2p2 <- lm(Price ~ H2p2, data=house)
lineal.simple.H8p2 <- lm(Price ~ H8p2, data=house)
lineal.simple.H10p1 <- lm(Price ~ H10p1, data=house)
lineal.simple.H13p1 <- lm(Price ~ H13p1, data=house)
lineal.simple.H18pA <- lm(Price ~ H18pA, data=house)
lineal.simple.H40p4 <- lm(Price ~ H40p4, data=house)

# Obtener información sobre los modelos
summary(lineal.simple.P1)
evaluate.rmse.house(lineal.simple.P1)

summary(lineal.simple.P5p1)
evaluate.rmse.house(lineal.simple.P5p1)

summary(lineal.simple.P6p2)
evaluate.rmse.house(lineal.simple.P6p2)

summary(lineal.simple.P11p4)
evaluate.rmse.house(lineal.simple.P11p4)

summary(lineal.simple.P14p9)
evaluate.rmse.house(lineal.simple.P14p9)

summary(lineal.simple.P15p1)
evaluate.rmse.house(lineal.simple.P15p1)

summary(lineal.simple.P15p3)
evaluate.rmse.house(lineal.simple.P15p3)

summary(lineal.simple.P16p2)
evaluate.rmse.house(lineal.simple.P16p2)

summary(lineal.simple.P18p2)
evaluate.rmse.house(lineal.simple.P18p2)

summary(lineal.simple.P27p4)
evaluate.rmse.house(lineal.simple.P27p4)

summary(lineal.simple.H2p2)
evaluate.rmse.house(lineal.simple.H2p2)

summary(lineal.simple.H8p2)
evaluate.rmse.house(lineal.simple.H8p2)

summary(lineal.simple.H10p1)
evaluate.rmse.house(lineal.simple.H10p1)

summary(lineal.simple.H13p1)
evaluate.rmse.house(lineal.simple.H13p1)

summary(lineal.simple.H18pA)
evaluate.rmse.house(lineal.simple.H18pA)

summary(lineal.simple.H40p4)
evaluate.rmse.house(lineal.simple.H40p4)

# Representación de los modelos elaborados
graf.simple.model.P1 <- ggplot(house, aes(y=Price, x=P1)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P1$coefficients[2],
              slope=lineal.simple.P1$coefficients[1],
              colour='deepskyblue',
              size=0.9)
graf.simple.model.P1

graf.simple.model.P5p1 <- ggplot(house, aes(y=Price, x=P5p1)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P5p1$coefficients[2],
              slope=lineal.simple.P5p1$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P5p1

graf.simple.model.P6p2 <- ggplot(house, aes(y=Price, x=P6p2)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P6p2$coefficients[2],
              slope=lineal.simple.P6p2$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P6p2

graf.simple.model.P11p4 <- ggplot(house, aes(y=Price, x=P11p4)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P11p4$coefficients[2],
              slope=lineal.simple.P11p4$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P11p4

graf.simple.model.P14p9 <- ggplot(house, aes(y=Price, x=P14p9)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P14p9$coefficients[2],
              slope=lineal.simple.P14p9$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P14p9

graf.simple.model.P15p1 <- ggplot(house, aes(y=Price, x=P15p1)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P15p1$coefficients[2],
              slope=lineal.simple.P15p1$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P15p1

graf.simple.model.P15p3 <- ggplot(house, aes(y=Price, x=P15p3)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P15p3$coefficients[2],
              slope=lineal.simple.P15p3$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P15p3

graf.simple.model.P16p2 <- ggplot(house, aes(y=Price, x=P16p2)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P16p2$coefficients[2],
              slope=lineal.simple.P16p2$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P16p2

graf.simple.model.P18p2 <- ggplot(house, aes(y=Price, x=P18p2)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P18p2$coefficients[2],
              slope=lineal.simple.P18p2$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P18p2

graf.simple.model.P27p4 <- ggplot(house, aes(y=Price, x=P27p4)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.P27p4$coefficients[2],
              slope=lineal.simple.P27p4$coefficients[1],
              colour='deepskyblue')
graf.simple.model.P27p4

graf.simple.model.H2p2 <- ggplot(house, aes(y=Price, x=H2p2)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H2p2$coefficients[2],
              slope=lineal.simple.H2p2$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H2p2

graf.simple.model.H8p2 <- ggplot(house, aes(y=Price, x=H8p2)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H8p2$coefficients[2],
              slope=lineal.simple.H8p2$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H8p2

graf.simple.model.H10p1 <- ggplot(house, aes(y=Price, x=H10p1)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H10p1$coefficients[2],
              slope=lineal.simple.H10p1$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H10p1

graf.simple.model.H13p1 <- ggplot(house, aes(y=Price, x=H13p1)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H13p1$coefficients[2],
              slope=lineal.simple.H13p1$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H13p1

graf.simple.model.H18pA <- ggplot(house, aes(y=Price, x=H18pA)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H18pA$coefficients[2],
              slope=lineal.simple.H18pA$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H18pA

graf.simple.model.H40p4 <- ggplot(house, aes(y=Price, x=H18pA)) +
  geom_point(colour='red', alpha=0.2) +
  geom_abline(intercept=lineal.simple.H40p4$coefficients[2],
              slope=lineal.simple.H40p4$coefficients[1],
              colour='deepskyblue')
graf.simple.model.H40p4

# Representar todas las gráficas
png('modelos_lineales.png', width=240*5, height = 480*5, res=120)
grid.arrange(graf.simple.model.P1, graf.simple.model.P5p1,
             graf.simple.model.P6p2, graf.simple.model.P11p4,
             graf.simple.model.P14p9, graf.simple.model.P15p1,
             graf.simple.model.P15p3, graf.simple.model.P16p2,
             graf.simple.model.P18p2, graf.simple.model.P27p4,
             graf.simple.model.H2p2, graf.simple.model.H8p2,
             graf.simple.model.H10p1, graf.simple.model.H13p1,
             graf.simple.model.H18pA, graf.simple.model.H40p4, nrow=8)
dev.off()

# Los 5 mejores modelos encontrados:
# 1º Price ~ P27p4
# 2º Price ~ P11p4
# 3º Price ~ H13p1
# 4º & Price ~ H40p4
# 5º & Price ~ P16p2

# Evaluar los modelos mediante validación cruzada 5-fold
run_k_fold <- function(i, x, model, type='lineal', tt = "test") {
  # Cargar conjuntos de entrenamiento
  file <- paste(x, "-5-", i, "tra.dat", sep=""); 
  x_tra <- read.house.dataset(file)
  
  # Cargar conjuntos de test
  file <- paste(x, "-5-", i, "tst.dat", sep="");
  x_tst <- read.house.dataset(file)
  
  if (tt == "train") { test <- x_tra }
  else { test <- x_tst }

  # Entrenar el modelo sobre el conjunto de train
  form <- terms(model)
  model.eval <- lm(formula=form, data=x_tra)

  # Evaluación del RMSE sobre test
  yprime=predict(model.eval, test)
  sqrt(sum(abs(test$Price-yprime)^2)/length(yprime)) ##RMSE
}

# Evaluación de todos los modelos
nombre <- './Datasets Regresion/house/house'
cat('Error RMSE del modelo P27p4 sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                            nombre, lineal.simple.P27p4), fill=T))
cat('Error RMSE del modelo P11p4 sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                            nombre, lineal.simple.P11p4), fill=T))
cat('Error RMSE del modelo H13p1 sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                            nombre, lineal.simple.H13p1), fill=T))
cat('Error RMSE del modelo H40p4 sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                            nombre, lineal.simple.H40p4), fill=T))
cat('Error RMSE del modelo P16p2 sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                            nombre, lineal.simple.P16p2), fill=T))

# Modelo lineal compuesto por todas las variables del dataset
lineal.model <- lm(Price~., data=house)

summary(lineal.model)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                  nombre, lineal.model), fill=T))

# Modelo lineal sin H2p2
lineal.model.fit1 <- lm(Price~.-H2p2, data=house)

summary(lineal.model.fit1)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit1), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit1), fill=T))

# Modelo no lineal con P27p4 al cuadrado
lineal.model.fit2 <- lm(Price~.-H2p2+I(P27p4^2), data=house)

summary(lineal.model.fit2)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit2), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit2), fill=T))

# Modelo no lineal con P27p4 elevado a 3
lineal.model.fit3 <- lm(Price~.-H2p2+I(P27p4^3), data=house)

summary(lineal.model.fit3)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit3), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit3), fill=T))

# Modelo no lineal con P27p4 elevado a 4
lineal.model.fit4 <- lm(Price~.-H2p2+I(P27p4^4), data=house)

summary(lineal.model.fit4)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit4), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                              nombre, lineal.model.fit4), fill=T))

# Modelo no lineal con P11p4 elevado al cuadrado
lineal.model.fit5 <- lm(Price~.-H2p2+I(P27p4^2)+I(P11p4^2), data=house)

summary(lineal.model.fit5)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit5), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit5), fill=T))

# Modelo no lineal con H13p1 elevado al cuadrado
lineal.model.fit6 <- lm(Price~.-H2p2+I(P27p4^2)+I(P11p4^2)+I(H13p1^2), data=house)

summary(lineal.model.fit6)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit6), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit6), fill=T))

# Modelo no lineal con H13p1 elevado al cuadrado y sin P11p4 al cuadrado
lineal.model.fit7 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2), data=house)

summary(lineal.model.fit7)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit7), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit7), fill=T))

# Modelo no lineal con H40p4 elevado al cuadrado al cuadrado
lineal.model.fit8 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2), data=house)

summary(lineal.model.fit8)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit8), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit8), fill=T))
# Modelo no lineal con P16p2 elevado al cuadrado al cuadrado
lineal.model.fit9 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2), data=house)

summary(lineal.model.fit9)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit9), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                              nombre, lineal.model.fit9), fill=T))

# Empezar con los otros

# Modelo no lineal con P1 elevado al cuadrado al cuadrado
lineal.model.fit10 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+I(P1^2), data=house)

summary(lineal.model.fit10)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit10), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit10), fill=T))

# Modelo no lineal con P5p1 elevado al cuadrado al cuadrado (No vale)
lineal.model.fit11 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P5p1^2), data=house)

summary(lineal.model.fit11)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit11), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit11), fill=T))

# Modelo no lineal con P11p4 elevado al cuadrado al cuadrado
lineal.model.fit12 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2), data=house)

summary(lineal.model.fit12)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit12), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                              nombre, lineal.model.fit12), fill=T))

# Modelo no lineal con P6p2 elevado al cuadrado al cuadrado (No vale)
lineal.model.fit13 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P6p2^2), data=house)

summary(lineal.model.fit13)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit13), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit13), fill=T))

# Modelo no lineal con H8p2 elevado al cuadrado
lineal.model.fit14 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(H8p2^2), data=house)

summary(lineal.model.fit14)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit14), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                                nombre, lineal.model.fit14), fill=T))

# Modelo no lineal con P18p2
lineal.model.fit15 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2), data=house) #I(H8p2^2)+

summary(lineal.model.fit15)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit15), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit15), fill=T))

# Modelo no lineal añadiendo H10p1
lineal.model.fit16 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2), data=house)

summary(lineal.model.fit16)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit16), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit16), fill=T))

# Modelo no lineal añadiendo H18pA al cuadrado
lineal.model.fit17 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2), data=house)

summary(lineal.model.fit17)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit17), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit17), fill=T))
# Modelo no lineal añadiendo H13p1 al cuadrado
lineal.model.fit18 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2)+I(H13p1^2), data=house)

summary(lineal.model.fit18)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit18), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit18), fill=T))

# Modelo no lineal añadiendo P14p9 al cuadrado
lineal.model.fit19 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2)+I(P14p9^2), data=house)

summary(lineal.model.fit19)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit19), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                          nombre, lineal.model.fit19), fill=T))

# Modelo no lineal con raíz de P18p2
lineal.model.fit20 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(sqrt(P18p2))+I(H10p1^2)+
                           I(H18pA^2), data=house)
summary(lineal.model.fit20)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit20), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                            nombre, lineal.model.fit20), fill=T))

# Modelo no lineal con raíz de H18pA
lineal.model.fit21 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2)+I(sqrt(P18p2))+sqrt(H18pA), data=house)
summary(lineal.model.fit21)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit21), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                                        nombre, lineal.model.fit21), fill=T))

# Modelo no lineal con raíz de P15p3
lineal.model.fit22 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2)+I(sqrt(P18p2))+sqrt(H18pA)+sqrt(P15p3), data=house)
summary(lineal.model.fit22)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit22), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                              nombre, lineal.model.fit22), fill=T))
# Modelo no lineal con raíz de P1
lineal.model.fit23 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^2)+
                           I(H18pA^2)+I(sqrt(P18p2))+sqrt(H18pA)+sqrt(P15p3)+sqrt(P1), data=house)
summary(lineal.model.fit23)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit23), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                                        nombre, lineal.model.fit23), fill=T))

# Modelo no lineal con H10p1 elevado a 4
lineal.model.fit24 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^4)+
                           I(H18pA^2)+I(sqrt(P18p2))+sqrt(H18pA)+sqrt(P15p3), data=house)
summary(lineal.model.fit24)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit24), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                                        nombre, lineal.model.fit24), fill=T))

lineal.model.fit25 <- lm(Price~.-H2p2+I(P27p4^2)+I(H13p1^2)+I(H40p4^2)+I(P16p2^2)+
                           I(P1^2)+I(P11p4^2)+I(P18p2^2)+I(H10p1^4)+
                           I(H18pA^2)+I(sqrt(P18p2))+sqrt(H18pA)+sqrt(P15p3)+
                           I(sqrt(P1*P15p3)), data=house)
summary(lineal.model.fit25)
cat('RMSE del modelo: ', evaluate.rmse.house(lineal.model.fit25), fill=T)
cat('Error RMSE del modelo lineal compuesto sobre 5-fold:', mean(sapply(1:5,run_k_fold,
                                                                        nombre, lineal.model.fit25), fill=T))
### Modelos k-NN

# Definición de la función para evaluar un modelo KNN
evaluate.knn_rmse.house <- function(model) {
  yprime = model$fitted.values
  sqrt(sum(abs(house$Price-yprime)^2)/length(yprime)) #Calcular RMSE
}

# Evaluar los modelos mediante validación cruzada 5-fold
run_knn_k_fold_house <- function(i, x, model, standarizate=NULL, tt = "test") {
  # Cargar conjuntos de entrenamiento
  file <- paste(x, "-5-", i, "tra.dat", sep=""); 
  x_tra <- read.csv(file, comment.char="@")
  
  # Cargar conjuntos de test
  file <- paste(x, "-5-", i, "tst.dat", sep="");
  x_tst <- read.csv(file, comment.char="@")
  
  names(x_tra) <- c('P1', 'P5p1', 'P6p2', 'P11p4', 'P14p9', 'P15p1', 'P15p3',
                    'P16p2', 'P18p2', 'P27p4', 'H2p2', 'H8p2', 'H10p1', 'H13p1',
                    'H18pA', 'H40p4', 'Price')
  names(x_tst) <- names(x_tra)
  
  # Estandarizar las variables proporcionadas
  for (i in standarizate) {
    # Aplicar sobre x_tra
    min.v <- min(x_tra[i])
    max.v <- max(x_tra[i])
    
    x_tra[[paste(i,'_rescaled',sep='')]] <- unlist((x_tra[i] - min.v)/(max.v-min.v))
    
    # Aplicar sobre x_tst
    min.v <- min(x_tst[i])
    max.v <- max(x_tst[i])
    
    x_tst[[paste(i,'_rescaled',sep='')]] <- unlist((x_tst[i] - min.v)/(max.v-min.v))
  }

  if (tt == "train") { test <- x_tra }
  else { test <- x_tst }
  
  # Entrenar el modelo sobre el conjunto de train
  form <- terms(model)
  model.eval <- kknn(formula=form, train=x_tra, test=test, k=ncol(model$CL),
                       distance=model$distance)
    
  # Evaluación del RMSE sobre test
  yprime=model.eval$fitted.values
  sqrt(sum(abs(test$Price-yprime)^2)/length(yprime)) ##RMSE
}

# Modelo por defecto
knn.model.fit1 <- kknn(Price~., house, house)

summary(knn.model.fit1)
cat('RMSE del modelo: ', evaluate.knn_rmse.house(knn.model.fit1), fill=T)
cat('Error RMSE del modelo kknn compuesto sobre 5-fold:', mean(sapply(1:5,run_knn_k_fold_house,
                                                            nombre, knn.model.fit1), fill=T))

# Reescalar P1 al intervalo [0,1]
min.p1 <- min(house$P1)
max.p1 <- max(house$P1)

house$P1_rescaled = (house$P1-min.p1)/(max.p1-min.p1)

# Entrenar otro modelo con P1 reescalado
knn.model.fit2 <- kknn(Price~.-P1, house, house)

summary(knn.model.fit2)
cat('RMSE del modelo: ', evaluate.knn_rmse.house(knn.model.fit2), fill=T)
cat('Error RMSE del modelo kknn compuesto sobre 5-fold:', mean(sapply(1:5,run_knn_k_fold_house,
                                                                      nombre, knn.model.fit2, 'P1'), fill=T))

# Entrenar otro modelo con transformaciuones para normalizar las variables
knn.model.fit3 <- kknn(Price~sqrt(P1_rescaled)+P5p1+I(P6p2^(1/3))+sqrt(P11p4)+sqrt(P14p9)+
                         I(P15p1^2)+I(P15p3^(1/3))+I(P16p2^2)+sqrt(P18p2)+sqrt(P27p4)+
                         sqrt(H2p2)+sqrt(H8p2)+I(H10p1^4)+sqrt(H13p1)+
                         sqrt(H18pA)+H40p4, house, house)

summary(knn.model.fit3)
cat('RMSE del modelo: ', evaluate.knn_rmse.house(knn.model.fit3), fill=T)
cat('Error RMSE del modelo kknn compuesto sobre 5-fold:', mean(sapply(1:5,run_knn_k_fold_house,
                                                                      nombre, knn.model.fit3, 'P1'), fill=T))

# Entrenar otro modelo con transformaciuones para normalizar las variables, pero sin reescalar P1
knn.model.fit4 <- kknn(Price~sqrt(P1)+P5p1+I(P6p2^(1/3))+sqrt(P11p4)+sqrt(P14p9)+
                         I(P15p1^2)+I(P15p3^(1/3))+I(P16p2^2)+sqrt(P18p2)+sqrt(P27p4)+
                         sqrt(H2p2)+sqrt(H8p2)+I(H10p1^4)+sqrt(H13p1)+
                         sqrt(H18pA)+H40p4, house, house)

summary(knn.model.fit4)
cat('RMSE del modelo: ', evaluate.knn_rmse.house(knn.model.fit3), fill=T)
cat('Error RMSE del modelo kknn compuesto sobre 5-fold:', mean(sapply(1:5,run_knn_k_fold_house,
                                                                      nombre, knn.model.fit4), fill=T))

# Evaluar el mejor valor de k
evaluate.knn <- function(k, model, standarizate=NULL) {
  # Se obtiene la formula del modelo
  form <- terms(model)
  
  # Evaluar el modelo
  model.eval <- kknn(form, house, house, k=k)
  meausures <- c(k, evaluate.knn_rmse.house(model.eval),
                 mean(sapply(1:5,run_knn_k_fold_house,
                  nombre, model.eval, standarizate)))
  names(meausures) <- c('k', 'RMSE', 'RMSE 5-fold')
  
  return(meausures)
}

# Valores de K usados en las pruebas
k.values <- c(1,3,7,13,21,51,75,103)

# Modelos evaluados sobre knn.model.fit4
normal.model.metrics <- sapply(k.values, evaluate.knn, knn.model.fit4)
normal.model.metrics

# Modelos evaluados sobre knn.model.fit1
original.model.metrics <- sapply(k.values, evaluate.knn, knn.model.fit1)
original.model.metrics
