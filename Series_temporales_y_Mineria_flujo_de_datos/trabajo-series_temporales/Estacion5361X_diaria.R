# Nicolás, Cubero
# Ejercicio autónomo I. Series temporales. Curso 2019-2020

library('tseries')
library('ggplot2')

# Cargar la serie diaria
met <- read.csv('Estacion5361X_diaria.txt', sep=';')
met$Fecha <- as.Date(met$Fecha, format='%Y-%m-%d')

# Dividir el conjunto de datos en train y test,
# (se reserva el 90 % para train y 10 % para test)
met.train <- met$Tmax[1:as.integer(0.9*length(met$Tmax))]
met.train.time <- met$Fecha[1:as.integer(0.9*length(met$Fecha))]

met.test <- met$Tmax[as.integer(0.9*length(met$Tmax)+1):length(met$Tmax)]
met.test.time <- met$Fecha[as.integer(0.9*length(met$Fecha)+1):length(met$Fecha)]

# Estudiar estacionalidad anual
met.anual <- ts(met[,2], frequency = 365)

# Descomponer la serie anualmente
desc.met.anual <- decompose(met.anual)

pdf('Tmax_diaria_decompose.pdf')
plot(desc.met.anual)
dev.off()

# Estimar estacionalidad
met.train.matrix <- matrix(data = met.train[1:((length(met.train)%/%365)*365)],
                           ncol = 365, byrow = TRUE)
estac.met.train <- apply(met.train.matrix, FUN=mean, MARGIN = 2)

rep.estac.met.train <- c(rep(estac.met.train,
                             length(met$Tmax)%/%length(estac.met.train)),
                         estac.met.train[1:(length(met$Tmax)%%length(estac.met.train))])

met.train.no_estac <- met.train - rep.estac.met.train[1:as.integer(0.9*length(rep.estac.met.train))]
met.test.no_estac <- met.test - rep.estac.met.train[as.integer(0.9*length(rep.estac.met.train)+1):length(rep.estac.met.train)]

# Representar la serie sin estacionalidad anual
pdf('Tmax_diaria_sin_estacionalidad_anual.pdf', width = 20)
plot(x = met.train.time, y = met.train.no_estac, type = "l",
     xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
     xlab='Meses', ylab='Tmax')
lines(met.test.time, met.test.no_estac, col='red')
dev.off()

# Representar el primer año
pdf('Tmax_diaria_sin_estacionalidad_anual_primer_ano.pdf', width = 20)
plot(x = met.train.time[1:365], y = met.train.no_estac[1:365], type = "l",
     xlim=c(met.train.time[1], met.train.time[1] + 365 ),
     xlab='Meses', ylab='Tmax')
dev.off()

# Test ADF para comprobar la estacionariedad de la serie
adf.test(met.train.no_estac)


# test ACF y test PACF de la serie con los residuos
pdf('Tmax_diaria_sin_estacionalidad_ACF.pdf')
acf(met.train.no_estac)
dev.off()

pdf('Tmax_diaria_sin_estacionalidad_PACF.pdf')
pacf(met.train.no_estac)
dev.off()

# Elaborar modelo ARIMA(1,0,0)
model.arima <- arima(met.train.no_estac, order = c(1,0,0))
model.arima

# Comprobar la normalidad de los residuos del modelo
Box.test(model.arima$residuals)
shapiro.test(model.arima$residuals)
jarque.bera.test(model.arima$residuals)

# Visualizar la distribución de los errores
pdf('Tmax_diaria_arima_1-0-0.pdf')
hist(model.arima$residuals, prob=T, col='firebrick1',
     main = 'Histograma de residuos del modelo ARIMA',
     border='white', xlab = 'Residuos', ylab='Densidad')
lines(density(model.arima$residuals), col='gray37')
dev.off()

# Predecir los conjuntos de train y test y evaluar el modelo
train.pred <- met.train.no_estac - model.arima$residuals # Train

arima.pred.test <- predict(model.arima, n.ahead = length(met.test.no_estac))
test.pred <- arima.pred.test$pred # Test

# Calcular los errores sobre el conjunto de train y test
sse.arima.train <- sum((model.arima$residuals)^2)
sse.arima.test <- sum((test.pred-met.test.no_estac)^2)

cat('Error SSE cometido sobre el conjunto de train: ', sse.arima.train, fill=T)
cat('Error SSE cometido sobre el conjunto de test: ', sse.arima.test, fill=T)

# Visualizar gráficamente la predición junto con los datos reales
pdf('Tmax_diaria_prediccion_sin_estacionalidad_arima-1-0-0.pdf', width = 20)
plot(x = met.train.time, y = met.train.no_estac,
     xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
     xlab='Meses', ylab='Serie sin estacionalidad', type = "l")
lines(met.test.time, met.test.no_estac, col='firebrick1')
lines(met.train.time, train.pred, col='cornflowerblue')
lines(met.test.time, test.pred, col='springgreen')
legend(x=24200, y=-2.10, legend = c('train - Valores reales', 'test - Valores reales',
                                    'train - Valores estimados', 'test - Valores estimados'),
       fill = c('black', 'firebrick1', 'cornflowerblue', 'springgreen'))
dev.off()

# Añadir estacionalidad sobre prediciones
train.pred.est <- train.pred + rep.estac.met.train[1:as.integer(0.9*length(rep.estac.met.train))]
test.pred.est <- test.pred + rep.estac.met.train[as.integer(0.9*length(rep.estac.met.train)+1):length(rep.estac.met.train)]

# Visualizar gráficamente la predición junto con los datos reales
pdf('Tmax_diaria_prediccion_arima-1-0-0.pdf', width = 20)
plot(x = met.train.time, y = met.train,
     xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
     xlab='Meses', ylab='Tmax', type = "l")
lines(met.test.time, met.test, col='firebrick1')
lines(met.train.time, train.pred.est, col='cornflowerblue')
lines(met.test.time, test.pred.est, col='springgreen')
legend(x=24160, y=27, legend = c('train - Valores reales', 'test - Valores reales',
                                 'train - Valores estimados', 'test - Valores estimados'),
       fill = c('black', 'firebrick1', 'cornflowerblue', 'springgreen'))
dev.off()

### Predecir toda la serie con el modelo ARIMA

# Eliminar estacionalidad anual de la serie completa
met.matrix <- matrix(data = met$Tmax[1:((length(met$Tmax)%/%365)*365)],
                     ncol = 365, byrow = TRUE)
estac.met <- apply(met.matrix, FUN=mean, MARGIN = 2)

rep.estac.met <- c(rep(estac.met,
                       length(met.anual)/length(estac.met)),
                   estac.met[1:(length(met.anual)%%length(estac.met))])

met.no_estac <- met$Tmax - rep.estac.met

# Ajustar modelo ARIMA (1,0,0)
model.arima.full <- arima(met.no_estac, order = c(1,0,0))
model.arima.full

# Predecir la siguiente semana
pred.arima <- predict(model.arima.full, n.ahead = 7)
pred <- pred.arima$pred

# Mostrar el error residual
cat('Error SSE cometido sobre los datos de entrenamiento: ',
    sum(pred.arima$se**2), fill = T)

# Añadir estacionalidad para el tramo predicho
pred.estac.met <- estac.met[(((1:7)+length(met$Tmax))%%length(estac.met)) + 1]
pred.estac <- pred + pred.estac.met

# Mostrar predición realizada
print(pred.estac)

# Mostrar gráfica con la predición
pdf('Tmax_predicion_primera_semana_marzo.pdf', width = 20)
plot(as.numeric(pred.estac), xlab='Días de Marzo de 2018',
     ylab='Tmax', type = "l", col = "firebrick1")
dev.off()

# Mostrar gráfica original junto con la predición
pdf('Tmax_predicion_primera_semana_marzo_serie_completa.pdf', width = 20)
plot(x = met$Fecha, y = met$Tmax,
     xlim=c(met$Fecha[1], met$Fecha[length(met$Fecha)]+2),
     xlab='Año', ylab='Tmax', type = "l", col = "gray37")
lines((met$Fecha[length(met$Fecha)]):(met$Fecha[length(met$Fecha)]+1),
      c(met$Tmax[length(met$Tmax)], pred.estac[1]), col='firebrick1', lty = "dashed")
lines((met$Fecha[length(met$Fecha)]+1):(met$Fecha[length(met$Fecha)]+7),
      pred.estac, col='firebrick1')
legend(x=241, y=27, legend = c('Serie original', 'Predición'),
       fill = c('gray37', 'firebrick1'))
dev.off()
