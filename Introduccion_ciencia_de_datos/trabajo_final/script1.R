###############################################################################
# Nombre del script: script1.R
# Desarrollado por: Nicolás Cubero
# Descripción: Script para el análisis exploratio del dataset house
# Nota: Este script ha sido desarrollado para ejecutarse de forma interactiva
#       línea por línea
###############################################################################

# Librerías cargadas
library('moments')
library('ggplot2')
library('GGally')
library('dplyr')
library('corrplot')

# Función para cargar house
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
  
# Obtener información sobre su estructura
str(house)
head(house)

# Obtener información sobre Missing Values
any(is.na(house))

# Determinar los estadísticos de posición:
# Valores mínimo y máximos, media, 1er cuartil, mediana, 3er cuartil
cat('Estadísticos de posición: Valores mínimo y máximos, media, 1er cuartil,',
    ' mediana, 3er cuartil', fill=T)
summary(house)

# Determinar los estadísticos de dispersión: desviación típica
cat('Desviación típica de los atributos', fill=T)
apply(house, MARGIN=2, FUN=sd)

# Determinar coeficientes de Skew y Kurtosis
cat('Coeficientes de Skew y Kurtosis', fill=T)
skew_kurtosis <- apply(house, MARGIN=2, FUN=function(x) {c(skewness(x),
                                                           kurtosis(x))})
rownames(skew_kurtosis) <- c('Skew', 'kurtosis')
skew_kurtosis

# Variable P1

# Diagrama boxplot
graf <- ggplot(house, aes(x='P1', y=P1)) +
        geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
        labs(cation='center', y='', x='')
graf

# Diagrama boxplot en escala logarítmica
hraf <- ggplot(house, aes(x='P1', y=P1)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  scale_y_log10() +
  labs(cation='center', y='', x='')
hraf

# Aislar la distribución comprendida entre [Q1-1.5*IQR, Q3+1.5*IQR]
quantiles.P1 <- quantile(house$P1)
iqr.P1 <- IQR(house$P1)

centro.house.P1 <- house %>% select(P1) %>%
  filter(P1 > (quantiles.P1[2]-1.5*iqr.P1) & (P1 < quantiles.P1[4]*1.5))

# Obtener estadísticos de posición y número de medidas
cat('Rango [Q1-1.5*IQR, Q3+1.5*IQR]:', fill=T)
centro.house.P1 %>% summarise(min=min(centro.house.P1$P1),
                                    q1=quantile(centro.house.P1$P1)[2],
                                    median=median(centro.house.P1$P1),
                                    mean=mean(centro.house.P1$P1),
                                    q3=quantile(centro.house.P1$P1)[4],
                                    max=max(centro.house.P1$P1),
                                    count=n(),
                                    ratio=n()/length(house$P1))

# Obtener los outliers inferiores a Q1-1.5*IQR
cat('Rango (-Inf, Q1-1.5*IQR):', fill=T)
low.outliers.house.P1 <- house %>% select(P1) %>%
  filter(P1 < (quantiles.P1[2]-1.5*iqr.P1))

# Obtener estadísticos de posición y número de medidas
low.outliers.house.P1 %>% summarise(min=min(low.outliers.house.P1),
                                    q1=quantile(low.outliers.house.P1)[2],
                                    median=median(low.outliers.house.P1),
                                    mean=mean(low.outliers.house.P1),
                                    q3=quantile(low.outliers.house.P1)[4],
                                    max=max(low.outliers.house.P1),
                                    count=n(),
                                    ratio=n()/length(house$P1))

# Obtener los outliers superiores a Q3+1.5*IQR
cat('Rango (Q3+1.5*IQR, +Inf):', fill=T)
upper.outliers.house.P1 <- house %>% select(P1) %>%
  filter(P1 > (quantiles.P1[4]+1.5*iqr.P1))

# Obtener estadísticos de posición y número de medidas
upper.outliers.house.P1 %>% summarise(min=min(upper.outliers.house.P1$P1),
                              q1=quantile(upper.outliers.house.P1$P1)[2],
                              median=median(upper.outliers.house.P1$P1),
                              mean=mean(upper.outliers.house.P1$P1),
                              q3=quantile(upper.outliers.house.P1$P1)[4],
                              max=max(upper.outliers.house.P1$P1),
                              count=n(),
                              ratio=n()/length(house$P1))

# Histrograma del centro de distribución
graf.high_density.P1 <- ggplot(centro.house.P1, aes(P1)) +
  geom_histogram(binwidth=15, color='red', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.high_density.P1

# Histograma del resto de la distribución
graf.upper_outliers.P1 <- ggplot(upper.outliers.house.P1, aes(P1)) +
  geom_histogram(binwidth=1000, colour='red') +
  labs(y='frecuencia absoluta')
graf.upper_outliers.P1

# Gráfica P5p1

# Diagrama boxplot
graf <- ggplot(house, aes(x='P5p1', y=P5p1)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P5p1)) +
  geom_histogram(binwidth=0.0025, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# Gráfica P6p1

# Diagrama boxplot
graf <- ggplot(house, aes(x='P6p1', y=P6p2)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P6p2)) +
  geom_histogram(binwidth=0.0025, color='gray58')
  labs(y='frecuencia absoluta')
graf

# Gráfica P11p4

# Diagrama boxplot
graf <- ggplot(house, aes(x='P11p4', y=P11p4)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P11p4)) +
  geom_histogram(binwidth=0.0025, color='gray58')
labs(y='frecuencia absoluta')
graf

# Gráfica P14p9

# Diagrama boxplot
graf <- ggplot(house, aes(x='P14p9', y=P14p9)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P14p9)) +
  geom_histogram(binwidth=0.00125, color='gray58')
labs(y='frecuencia absoluta')
graf

# Gráfica P15p1

# Diagrama boxplot
graf <- ggplot(house, aes(x='P15p1', y=P15p1)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P15p1)) +
  geom_histogram(binwidth=0.00064, color='gray58')
labs(y='frecuencia absoluta')
graf

# Gráfica P15p3

# Diagrama boxplot
graf <- ggplot(house, aes(x='P15p3', y=P15p3)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P15p3)) +
  geom_histogram(binwidth=0.00064, color='gray58')
labs(y='frecuencia absoluta')
graf

# Gráfica P16p2

# Diagrama boxplot
graf <- ggplot(house, aes(x='P16p2', y=P16p2)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P16p2)) +
  geom_histogram(binwidth=0.00125, color='gray58')
labs(y='frecuencia absoluta')
graf

# P18p2

# Diagrama boxplot
graf <- ggplot(house, aes(x='P18p2', y=P18p2)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P18p2)) +
  geom_histogram(binwidth=0.000125, color='gray58') +
labs(y='frecuencia absoluta')
graf

# Histograma del centro de la distribución de P18p2
iqr.P18p2 <- IQR(house$P18p2)
quantiles.P18p2 <- quantile(house$P18p2)

graf <- ggplot(house, aes(P18p2)) +
  geom_histogram(binwidth=0.0001, color='gray58') +
  labs(y='frecuencia absoluta') +
  xlim(quantiles.P18p2[2]-1.5*iqr.P18p2, quantiles.P18p2[4]+1.5*iqr.P18p2)
graf

# P27p4

# Diagrama boxplot
graf <- ggplot(house, aes(x='P27p4', y=P27p4)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(P27p4)) +
  geom_histogram(binwidth=0.000125, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H2p2

# Diagrama boxplot
graf <- ggplot(house, aes(x='H2p2', y=H2p2)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H2p2)) +
  geom_histogram(binwidth=0.0005, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H8p2

# Diagrama boxplot
graf <- ggplot(house, aes(x='H8p2', y=H8p2)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H8p2)) +
  geom_histogram(binwidth=0.0005, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H10p1

# Diagrama boxplot
graf <- ggplot(house, aes(x='H10p1', y=H10p1)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H10p1)) +
  geom_histogram(binwidth=0.0005, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H13p1

# Diagrama boxplot
graf <- ggplot(house, aes(x='H13p1', y=H13p1)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H13p1)) +
  geom_histogram(binwidth=0.0005, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H18pA

# Diagrama boxplot
graf <- ggplot(house, aes(x='H18pA', y=H18pA)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H18pA)) +
  geom_histogram(binwidth=0.0025, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# H40p4

# Diagrama boxplot
graf <- ggplot(house, aes(x='H40p4', y=H40p4)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(H40p4)) +
  geom_histogram(binwidth=0.0025, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# Price

# Diagrama boxplot
graf <- ggplot(house, aes(x='Price', y=Price)) +
  geom_boxplot(outlier.alpha=0.1, outlier.colour='red', outlier.shape=1) +
  labs(cation='center', y='', x='')
graf

# Histograma
graf <- ggplot(house, aes(Price)) +
  geom_histogram(binwidth=200, color='gray58') +
  labs(y='frecuencia absoluta')
graf

# Estudiar los pares de relaciones
png('relaciones_pares_variables.png', width=480*5, height = 240*5, res=120)
ggpairs(house, mapping=ggplot2::aes(colour='red', alpha=0.05),
          upper = list(continuous = "cor", corMethod = "kendall"))
dev.off()

# Estudiar la relación entre las variables P1 y P5p1
graf.p1.p5p1 <- ggplot(house, aes(x=P1, y=P5p1)) +
  geom_point(color='red', alpha=0.3)
graf.p1.p5p1

# Dibujar un gráfico de correlaciones basados en Spearman
corr <- cor(house, method='kendall')

pdf('Correlograma_kendall_house.pdf', width=14, height = 7)
par(mfrow=c(1,2))
corrplot(corr, method='ellipse', type='lower')
corrplot(corr, method='number', type='lower', number.cex = 0.80)
dev.off()
