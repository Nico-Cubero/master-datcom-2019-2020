###############################################################################
# Nombre del script: script4.R
# Desarrollado por: Nicolás Cubero
# Descripción: Script para el análisis exploratorio del dataset vehicle
# 
# Nota: Este script ha sido desarrollado para ejecutarse de forma interactiva
#       línea por línea
###############################################################################
# Librerías cargadas
library('moments')
library('ggplot2')
library('dplyr')
library('GGally')
library('corrplot')

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


# Comprobar la existencia de Missing Values
any(is.na(vehicle))

# Analizar brevemente su estructura
str(vehicle)
head(vehicle)

# Determinar las distribuciones de las variables
# Determinar los estadísticos de posición:
# Valores mínimo y máximos, media, 1er cuartil, mediana, 3er cuartil
cat('Estadísticos de posición: Valores mínimo y máximos, media, 1er cuartil,',
    ' mediana, 3er cuartil', fill=T)
summary(vehicle[,c(1:ncol(vehicle)-1)])

# Determinar los estadísticos de dispersión: desviación típica
cat('Desviación típica de los atributos', fill=T)
apply(vehicle[,c(1:ncol(vehicle)-1)], MARGIN=2, FUN=sd)

# Determinar coeficientes de Skew y Kurtosis
cat('Coeficientes de Skew y Kurtosis', fill=T)
skew_kurtosis <- apply(vehicle[,c(1:ncol(vehicle)-1)], MARGIN=2,
                       FUN=function(x) {c(skewness(x), kurtosis(x))})
rownames(skew_kurtosis) <- c('Skew', 'kurtosis')
skew_kurtosis

# Analizar las distribuciones de las variables
# Compactness
# Diagrama boxplot
graf.compactness.boxplot <- ggplot(vehicle, aes(x='Compactness', y=Compactness)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.compactness.boxplot

# Histograma
graf.compactness.histogram <- ggplot(vehicle, aes(Compactness, fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.compactness.histogram

# Estudiamos normalidad con el test de Shapiro-Milk
shapiro.test(vehicle$Compactness)

# Circularity
# Diagrama boxplot
graf.circularity.boxplot <- ggplot(vehicle, aes(x='Circularity', y=Circularity)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.circularity.boxplot

# Histograma
graf.circularity.histogram <- ggplot(vehicle, aes(Circularity, fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.circularity.histogram

# Estudiamos normalidad con el test de Shapiro-Milk
shapiro.test(vehicle$Circularity)

# Distance_circularity
# Diagrama boxplot
graf.distance_circularity.boxplot <- ggplot(vehicle, aes(x='Distance_circularity',
                                                y=Distance_circularity)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.distance_circularity.boxplot

# Histograma
graf.distance_circularity.histogram <- ggplot(vehicle,
                                      aes(Distance_circularity, fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.distance_circularity.histogram

# Estudiamos normalidad con el test de Shapiro-Milk
shapiro.test(vehicle$Distance_circularity)

# Radius_ratio
# Diagrama boxplot
graf.radius_ratio.boxplot <- ggplot(vehicle, aes(x='Radius_ratio',
                                                         y=Radius_ratio)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.radius_ratio.boxplot

# Histograma
graf.radius_ratio.histogram <- ggplot(vehicle,
                                              aes(Radius_ratio, fill=Class)) +
  geom_histogram(binwidth=3, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.radius_ratio.histogram

# Calcular rango de valores pertenecientes a la distribución
iqr.radious_ratio <- IQR(vehicle$Radius_ratio)
quantiles.radious_ratio <- quantile(vehicle$Radius_ratio)

q1.radious_ratio <- quantiles.radious_ratio[2]
q3.radious_ratio <- quantiles.radious_ratio[4]

cat('Intervalo de distribución: [', q1.radious_ratio-1.5*iqr.radious_ratio,',',
    q3.radious_ratio+1.5*iqr.radious_ratio,']', fill=T)

# Aislamos los outliers que se encuentran por encima del límite superior
vehicle.outliers.radious_ratio <- vehicle %>% filter(Radius_ratio>276)
vehicle.outliers.radious_ratio %>% select(Radius_ratio, Class)

# Estudiamos normalidad con el test de Shapiro-Milk
shapiro.test(vehicle$Radius_ratio)

# Praxis_aspect_ratio
# Diagrama boxplot
pdf('Praxis_aspect_ratio_boxplot.pdf')
graf.praxis_aspect_ratio.boxplot <- ggplot(vehicle,
                                               aes(x='Praxis_aspect_ratio',
                                                   y=Praxis_aspect_ratio)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.praxis_aspect_ratio.boxplot
dev.off()

# Histograma
pdf('Praxis_aspect_ratio_histograma.pdf')
graf.praxis_aspect_ratio.histogram <- ggplot(vehicle,
                                                 aes(Praxis_aspect_ratio, fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.praxis_aspect_ratio.histogram
dev.off()

# Max_length_aspect_ratio
# Diagrama boxplot
graf.max_length_aspect_ratio.boxplot <- ggplot(vehicle,
                                              aes(x='Max_length_aspect_ratio',
                                             y=Max_length_aspect_ratio)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.max_length_aspect_ratio.boxplot

# Histograma
graf.max_length_aspect_ratio.histogram <- ggplot(vehicle,
                                    aes(Max_length_aspect_ratio, fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.max_length_aspect_ratio.histogram

# Scatter_ratio
# Diagrama boxplot
pdf('Scatter_ratio_boxplot.pdf')
graf.scatter_ratio.boxplot <- ggplot(vehicle, aes(x='Scatter_ratio',
                                                   y=Scatter_ratio)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.scatter_ratio.boxplot
dev.off()

# Histograma
pdf('Scatter_ratio_histograma.pdf')
graf.scatter_ratio.histogram <- ggplot(vehicle, aes(Scatter_ratio,
                                                    fill=Class)) +
  geom_histogram(binwidth=2, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.scatter_ratio.histogram
dev.off()

# Elongatedness
# Diagrama boxplot
pdf('Elongatedness_boxplot.pdf')
graf.elongatedness.boxplot <- ggplot(vehicle, aes(x='Elongatedness',
                                                  y=Elongatedness)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.elongatedness.boxplot
dev.off()

# Histograma
pdf('Elongatedness_histograma.pdf')
graf.elongatedness.histogram <- ggplot(vehicle, aes(Elongatedness,
                                                    fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.elongatedness.histogram
dev.off()

# Praxis_rectangular
# Diagrama boxplot
pdf('Praxis_rectangular_boxplot.pdf')
graf.praxis_rectangular.boxplot <- ggplot(vehicle, aes(x='Praxis_rectangular',
                                                  y=Praxis_rectangular)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.praxis_rectangular.boxplot
dev.off()

# Histograma
pdf('Praxis_rectangular_histograma.pdf')
graf.praxis_rectangular.histogram <- ggplot(vehicle, aes(Praxis_rectangular,
                                                    fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.praxis_rectangular.histogram
dev.off()

# Length_rectangular
# Diagrama boxplot
pdf('Length_rectangular_boxplot.pdf')
graf.length_rectangular.boxplot <- ggplot(vehicle, aes(x='Length_rectangular',
                                                       y=Length_rectangular)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.length_rectangular.boxplot
dev.off()

# Histograma
pdf('Length_rectangular_histograma.pdf')
graf.length_rectangular.histogram <- ggplot(vehicle, aes(Length_rectangular,
                                                         fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.length_rectangular.histogram
dev.off()

# Major_variance
# Diagrama boxplot
pdf('Major_variance_boxplot.pdf')
graf.major_variance.boxplot <- ggplot(vehicle, aes(x='Major_variance',
                                                       y=Major_variance)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.major_variance.boxplot
dev.off()

# Histograma
pdf('Major_variance_histograma.pdf')
graf.major_variance.histogram <- ggplot(vehicle, aes(Major_variance,
                                                         fill=Class)) +
  geom_histogram(binwidth=3, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.major_variance.histogram
dev.off()

# Analizar outliers por encima del valor 300
vehicle.outliers.major_variance <- vehicle %>% filter(Major_variance>300)
vehicle.outliers.major_variance %>% select(Major_variance, Class)

# Minor_variance
# Diagrama boxplot
pdf('Minor_variance_boxplot.pdf')
graf.minor_variance.boxplot <- ggplot(vehicle, aes(x='Minor_variance',
                                                   y=Minor_variance)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.minor_variance.boxplot
dev.off()

# Histograma
pdf('Minor_variance_histograma.pdf')
graf.minor_variance.histogram <- ggplot(vehicle, aes(Minor_variance,
                                                     fill=Class)) +
  geom_histogram(binwidth=15, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.minor_variance.histogram
dev.off()

# Gyration_radius
# Diagrama boxplot
pdf('Gyration_radius_boxplot.pdf')
graf.gyration_radius.boxplot <- ggplot(vehicle, aes(x='Gyration_radius',
                                                   y=Gyration_radius)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.gyration_radius.boxplot
dev.off()

# Histograma
pdf('Gyration_radius_histograma.pdf')
graf.gyration_radius.histogram <- ggplot(vehicle, aes(Gyration_radius,
                                                     fill=Class)) +
  geom_histogram(binwidth=3, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.gyration_radius.histogram
dev.off()

# Major_skewness
# Diagrama boxplot
pdf('Major_skewness_boxplot.pdf')
graf.major_skewness.boxplot <- ggplot(vehicle, aes(x='Major_skewness',
                                                    y=Major_skewness)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.major_skewness.boxplot
dev.off()

# Histograma
pdf('Major_skewness_histograma.pdf')
graf.major_skewness.histogram <- ggplot(vehicle, aes(Major_skewness,
                                                      fill=Class)) +
  geom_histogram(binwidth=2, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.major_skewness.histogram
dev.off()

# Minor_skewness
# Diagrama boxplot
pdf('Minor_skewness_boxplot.pdf')
graf.minor_skewness.boxplot <- ggplot(vehicle, aes(x='Minor_skewness',
                                                   y=Minor_skewness)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.minor_skewness.boxplot
dev.off()

# Histograma
pdf('Minor_skewness_histograma.pdf')
graf.minor_skewness.histogram <- ggplot(vehicle, aes(Minor_skewness,
                                                     fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.minor_skewness.histogram
dev.off()

# Minor_kurtosis
# Diagrama boxplot
pdf('Minor_kurtosis_boxplot.pdf')
graf.minor_kurtosis.boxplot <- ggplot(vehicle, aes(x='Minor_kurtosis',
                                                   y=Minor_kurtosis)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.minor_kurtosis.boxplot
dev.off()

# Histograma
pdf('Minor_kurtosis_histograma.pdf')
graf.minor_kurtosis.histogram <- ggplot(vehicle, aes(Minor_kurtosis,
                                                     fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.minor_kurtosis.histogram
dev.off()

# Major_kurtosis
# Diagrama boxplot
pdf('Major_kurtosis_boxplot.pdf')
graf.major_kurtosis.boxplot <- ggplot(vehicle, aes(x='Major_kurtosis',
                                                   y=Major_kurtosis)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.major_kurtosis.boxplot
dev.off()

# Histograma
pdf('Major_kurtosis_histograma.pdf')
graf.major_kurtosis.histogram <- ggplot(vehicle, aes(Major_kurtosis,
                                                     fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.major_kurtosis.histogram
dev.off()

# Major_kurtosis
# Diagrama boxplot
pdf('Hollows_ratio_boxplot.pdf')
graf.hollows_ratio.boxplot <- ggplot(vehicle, aes(x='Hollows_ratio',
                                                   y=Hollows_ratio)) +
  geom_boxplot(outlier.alpha=0.4, outlier.colour='red', outlier.shape=8) +
  labs(cation='center', y='', x='')
graf.hollows_ratio.boxplot
dev.off()

# Histograma
pdf('Hollows_ratio_histograma.pdf')
graf.hollows_ratio.histogram <- ggplot(vehicle, aes(Hollows_ratio,
                                                     fill=Class)) +
  geom_histogram(binwidth=1, color='black', alpha=0.4) +
  labs(y='frecuencia absoluta')
graf.hollows_ratio.histogram
dev.off()

# Class
# Analizar la distribución de los valores
table(vehicle$Class)

# Representar gráficamente esta información
pdf('Class_barplot.pdf')
graf.class <- ggplot(vehicle, aes(x=Class, fill=Class)) +
  geom_bar()
graf.class
dev.off()

# Representar diagramas scatterplot entre los pares de variables
png('relaciones_variables_vehicle.png', width=480*5, height = 240*5, res=120)
ggpairs(vehicle[,1:ncol(vehicle)-1], mapping=ggplot2::aes(colour='red', alpha=0.05),
        upper = list(corMethod = "kendall"))
dev.off()

# Dibujar un gráfico de correlaciones basados en Spearman
corr <- cor(vehicle[,1:ncol(vehicle)-1], method='kendall')

pdf('Correlograma_spearman_vehicle.pdf', width=14, height = 7)
par(mfrow=c(1,2))
corrplot(corr, method='ellipse', type='lower')
corrplot(corr, method='number', type='lower', number.cex = 0.80)
dev.off()
