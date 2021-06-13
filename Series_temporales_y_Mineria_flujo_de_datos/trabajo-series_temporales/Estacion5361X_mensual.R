# Nicolás, Cubero
# Ejercicio autónomo I. Series temporales. Curso 2019-2020

library('tseries')
library('ggplot2')

# Cargar la serie mensual
met <- read.csv('Estacion5361X_mensual.txt', sep=';')
met.serie <- ts(met[,2], start = c(2013, 5), end = c(2018, 2),
                deltat = 1/12, frequency = 12)

# Representar gráficamente la serie
plot.ts(met.serie)

# Dividir el conjunto de datos en train y test,
# (se reserva el 90 % para train y 10 % para test)
met.train <- met.serie[1:as.integer(0.9*length(met.serie))]
met.train.time <- met$Mes[1:as.integer(0.9*length(met$Mes))]
met.train.serie <- ts(met.train, start = c(2013, 5),
                      deltat = 1/12, frequency = 12)

met.test <- met.serie[as.integer(0.9*length(met.serie)+1):length(met.serie)]
met.test.time <- met$Mes[as.integer(0.9*length(met$Mes)+1):length(met$Mes)]
met.test.serie <- ts(met.test, end = c(2018, 2),
                      deltat = 1/12, frequency = 12)

# Descomposición por defecto del método decompose
met.default.desc <- decompose(met.train.serie)
plot(met.default.desc, xlab = 'Tiempo')

# Estimar estacionalidad y eliminarla
met.train.matrix <- matrix(data = met.train[1:((length(met.train)%/%12)*12)],
                           ncol = 12, byrow = TRUE)
estac.met.train <- apply(met.train.matrix, FUN=mean, MARGIN = 2)

rep.estac.met.train <- c(rep(estac.met.train,
                           length(met.serie)/length(estac.met.train)),
                         estac.met.train[1:(length(met.serie)%%length(estac.met.train))])

met.train.no_estac <- met.train - rep.estac.met.train[1:as.integer(0.9*length(rep.estac.met.train))]
met.test.no_estac <- met.test - rep.estac.met.train[as.integer(0.9*length(rep.estac.met.train)+1):length(rep.estac.met.train)]

# Dibujar serie sin estacionalidad
pdf('Tmax_mensual_sin_estacionalidad.pdf')
plot(x = met.train.time, y = met.train.no_estac, type = "l",
        xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
        xlab='Meses', ylab='Tmax')
lines(met.test.time, met.test.no_estac, col='firebrick1')
dev.off()

# Test ADF para comprobar que la serie es estacionaria
adf.test(met.train.no_estac)

# Gráfico ACF
pdf('Tmax_mensual_sin_estacionalidad_ACF.pdf')
acf(met.train.no_estac)
dev.off()

# Gráfico PACF
pdf('Tmax_mensual_sin_estacionalidad_PACF.pdf')
pacf(met.train.no_estac)
dev.off()

# Probar modelo ARIMA(1,0,0)
model.arima <- arima(met.train.no_estac, order = c(1,0,0))
model.arima

# Comprobar que los residuos son normales
Box.test(model.arima$residuals)
shapiro.test(model.arima$residuals)
jarque.bera.test(model.arima$residuals)

# Visualizar la distribución de los errores
pdf('Tmax_mensual_arima_1-0-0.pdf')
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
pdf('Tmax_prediccion_sin_estacionalidad_arima-1-0-0.pdf', width = 10)
plot(x = met.train.time, y = met.train.no_estac,
        xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
        xlab='Meses', ylab='Serie sin estacionalidad', type = "l")
lines(met.test.time, met.test.no_estac, col='firebrick1')
lines(met.train.time, train.pred, col='cornflowerblue')
lines(met.test.time, test.pred, col='springgreen')
legend(x=35, y=-2.10, legend = c('train - Valores reales', 'test - Valores reales',
                              'train - Valores estimados', 'test - Valores estimados'),
       fill = c('gray37', 'firebrick1', 'cornflowerblue', 'springgreen'))
dev.off()

# Añadir estacionalidad sobre prediciones
train.pred.est <- train.pred + rep.estac.met.train[1:as.integer(0.9*length(rep.estac.met.train))]
test.pred.est <- test.pred + rep.estac.met.train[as.integer(0.9*length(rep.estac.met.train)+1):length(rep.estac.met.train)]

# Visualizar gráficamente la predición junto con los datos reales
pdf('Tmax_prediccion_arima-1-0-0.pdf', width = 10)
plot(x = met.train.time, y = met.train,
     xlim=c(met.train.time[1], met.test.time[length(met.test.time)]),
     xlab='Meses', ylab='Tmax', type = "l")
lines(met.test.time, met.test, col='firebrick1')
lines(met.train.time, train.pred.est, col='cornflowerblue')
lines(met.test.time, test.pred.est, col='springgreen')
legend(x=35, y=27, legend = c('train - Valores reales', 'test - Valores reales',
                                    'train - Valores estimados', 'test - Valores estimados'),
       fill = c('gray37', 'firebrick1', 'cornflowerblue', 'springgreen'))
dev.off()

### Predecir toda la serie con el modelo ARIMA

# Eliminar estacionalidad de la serie
met.matrix <- matrix(data = met$Tmax[1:((length(met$Tmax)%/%12)*12)],
                           ncol = 12, byrow = TRUE)
estac.met <- apply(met.matrix, FUN=mean, MARGIN = 2)

rep.estac.met <- c(rep(estac.met,
                        length(met.serie)/length(estac.met)),
                        estac.met[1:(length(met.serie)%%length(estac.met))])

met.no_estac <- met$Tmax - rep.estac.met

# Ajustar modelo ARIMA (1,0,0)
model.arima.full <- arima(met.no_estac, order = c(1,0,0))
model.arima.full

# Predecir los meses 59 y 60 (los dos meses siguientes)
pred.arima <- predict(model.arima.full, n.ahead = 2)
pred <- pred.arima$pred

# Mostrar el error residual
cat('Error SSE cometido sobre los datos de entrenamiento: ',
    sum(pred.arima$se**2), fill = T)

# Añadir estacionalidad para el tramo predicho
pred.estac.met <- estac.met[(((59:60)-met$Mes[1])%%12)+1]
pred.estac <- pred + pred.estac.met

# Mostrar predición realizada
print(pred.estac)

# Mostrar gráfica original junto con la predición
pdf('Tmax_predicion_marzo_abril.pdf', width = 10)
plot(x = met$Mes, y = met$Tmax,
     xlim=c(met$Mes[1], met$Mes[length(met$Mes)]+2),
     xlab='Año', ylab='Tmax', type = "l", col = "gray37", xaxt="n")
axis(1, at = (2013:2018)*12, labels = 2013:2018)
lines((met$Mes[length(met$Mes)]):(met$Mes[length(met$Mes)]+1),
      c(met$Tmax[length(met$Tmax)], pred.estac[1]), col='firebrick1')
lines((met$Mes[length(met$Mes)]+1):(met$Mes[length(met$Mes)]+2),
      pred.estac, col='firebrick1')
legend(x=4, y=24, legend = c('Serie original', 'Predición'),
       fill = c('gray37', 'firebrick1'))
dev.off()
