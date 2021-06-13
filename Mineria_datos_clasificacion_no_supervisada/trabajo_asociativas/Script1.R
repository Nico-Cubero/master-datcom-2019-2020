# Librerías importados
library('arules')
library('ggplot2')

# Función para lectura del dataset
read.statlog.heart.dataset <- function(filename) {
  dat <- read.csv(filename, sep=' ')
  
  colnames(dat) <- c('age', 'sex', 'chest pain type',
                     'resting blood pressure', 'serum cholestoral',
                     'fasting blood sugar', 
                     'resting electrocardiographic results',
                     'maximum heart rate achieved',
                     'exercise induced angina', 'oldpeak',
                     'slope of the peak exercise ST segment',
                     'number of major vessels', 'thal', 'heart dissease')

  # Preprocesar el dataset para asignar los tipos de datos correctos
  # Convertir age en integer
  dat$age <- as.integer(dat$age)
  
  # Convertir sex en factor
  dat$sex <- factor(x=dat$sex, levels=c(1,0), labels=c('male','female'))
  
  # Convertir chest pain type en factor
  dat[,3] <- factor(dat[,3], levels=1:4, labels=c('typical angina',
                                                  'atypical angina',
                                                  'non-anginal pain',
                                                  'asymptomatic'))

  # Convertir fasting blood sugar en factor binario
  dat[,6] <- as.logical(dat[,6])
  
  # Convertir resting electrocardiographic results en factores
  dat[,7] <- factor(dat[,7], levels=0:2, labels=c('Normal',
                                              'ST-T wave anormality',
                                              'left ventricular hypertrophy'))
  
  # Convertir exercise induced angina en factores binarios
  dat[,9] <- as.logical(dat[,9])
  
  # Convertir the slope of the peak exercise ST segment en factores
  dat[,11] <- factor(dat[,11], levels=1:3, labels=c('Upsloping',
                                                       'Flat',
                                                       'Downsloping'))
  
  # Convertir number of major vessels
  dat[,12] <- as.integer(dat[,12])
  
  # Convertir thal en factores
  dat[,13] <- factor(dat[,13], levels=c(3,6,7), labels=c('Normal',
                                                        'Fixed defect',
                                                        'Reversable defect'))
  # Convertir la clase en factor binaria
  dat[,14] <- dat[,14] == 2.0
  
  return(dat)
}

# Cargar dataset
heart <- read.statlog.heart.dataset('./heart.dat')

# Información breve del dataset
head(heart)
summary(heart)

# Estudiar los estadísticos de posición de age
summary(heart$age)

# Discretización del atributo age
heart[['age']] <- ordered(cut(heart[['age']],
                             c(29,60,+Inf),
                             labels=c('Adult', 'Elderly'),
                             right = F))

# Estudiar los estadísticos de posición resting blood pressure
summary(heart[['resting blood pressure']])

# Discretizar el atributo resting blood presssure
heart[['resting blood pressure']] <- ordered(
                cut(heart[['resting blood pressure']],
                    c(94,120,130,140,+Inf),
                    labels=c('Normal', 'Elevated',
                             'Hypertension-stage1',
                             'Hypertension-stage2'),
                    right = F))

# Estudiar los estadísticos de posición de serum cholesterol
summary(heart[['serum cholestoral']])

# Discretizar el atributo serum cholesterol
heart[['serum cholestoral']] <- ordered(
          cut(heart[['serum cholestoral']],
              c(126,200,240,+Inf),
              labels=c('Normal level',
                       'High level',
                       'Dangerous level'),
              right = F))

# Estudiar los estadísticos de posición de maximum heart rate achieved
summary(heart[['maximum heart rate achieved']])

# Estudiar gráficamente el atributo "maximum heart rate achieved"
pdf('maximum_heart_rate_achieved.pdf')
graf <- ggplot2::ggplot(data=heart, aes(x=`maximum heart rate achieved`)) +
  ggplot2::geom_histogram(binwidth = 2, colour='white',
                          fill='coral3') +
  ggplot2::ylab('Frecuencia absoluta')
graf
dev.off()

# Discretizar en 4 intervalos
heart[['maximum heart rate achieved']] <- discretize(
  heart[['maximum heart rate achieved']],
  method = 'frequency', breaks=4)

# Estudiar los estadísticos de posición de oldpeek
summary(heart[['oldpeak']])

# Estudiar gráficamente el atributo "oldpeak"
pdf('oldpeak.pdf')
graf <- ggplot2::ggplot(data=heart, aes(x=oldpeak)) +
  ggplot2::geom_histogram(binwidth = 0.1, colour='white',
                          fill='coral3') +
  ggplot2::ylab('Frecuencia absoluta')
graf
dev.off()

# Discretizar el atributo oldpeak en 3 intervalos de igual frecuencia
heart[['oldpeak']] <- discretize(
  heart[['oldpeak']], method = 'frequency',
  breaks=3)

# Convertir exercise induced angina y heart dissease en factores
heart[['exercise induced angina']] <- factor(
  heart[['exercise induced angina']], ordered = TRUE)
heart[['heart dissease']] <- factor(
  heart[['heart dissease']], ordered = TRUE)
