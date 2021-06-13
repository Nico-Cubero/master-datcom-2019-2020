# Librerías importados
library('arules')
library('ggplot2')
library('arulesViz')

# Función para lectura del dataset
read.statlog.heart.transactions <- function(filename) {
  dat <- read.table(filename, sep=' ')
  
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
  dat[,12] <- as.factor(as.integer(dat[,12]))
  
  # Convertir thal en factores
  dat[,13] <- factor(dat[,13], levels=c(3,6,7), labels=c('Normal',
                                               'Fixed defect',
                                               'Reversable defect'))
  # Convertir la clase en factor binaria
  dat[,14] <- dat[,14] == 2.0

  # Discretización del atributo age
  dat[['age']] <- ordered(cut(dat[['age']],
                                c(29,60,+Inf),
                                labels=c('Adult', 'Elderly'),
                                right = F))
  
  # Discretizar el atributo resting blood presssure
  dat[['resting blood pressure']] <- ordered(
    cut(dat[['resting blood pressure']],
        c(94,120,130,140,+Inf),
        labels=c('Normal', 'Elevated',
                 'Hypertension-stage1',
                 'Hypertension-stage2'),
        right = F))
  
  # Discretizar el atributo serum cholesterol
  dat[['serum cholestoral']] <- ordered(
    cut(dat[['serum cholestoral']],
        c(126,200,240,+Inf),
        labels=c('Normal level',
                 'High level',
                 'Dangerous level'),
        right = F))
  
  # Discretizar en 4 intervalos
  dat[['maximum heart rate achieved']] <- discretize(
    dat[['maximum heart rate achieved']], method = 'frequency',
    breaks=4)
  
  # Discretizar el atributo oldpeak en 3 intervalos de igual frecuencia
  dat[['oldpeak']] <- discretize(
    dat[['oldpeak']], method = 'frequency',
    breaks=3)

  # Convertir exercise induced angina y heart dissease en factores
  dat[['exercise induced angina']] <- factor(
    dat[['exercise induced angina']], ordered = TRUE)
  dat[['heart dissease']] <- factor(
    dat[['heart dissease']], ordered = TRUE)
  
  # Convertir el dataset en un conjunto de transacciones
  dat <- as(dat, 'transactions')
  
  return(dat)
}

# Definir función para eliminar reglas redundantes
remove.redundat.rules <- function(rule.set) {
  
  # Identificar las reglas que son superconjuntos de otras
  subset.rules.mask <- is.subset(rule.set)
  redundant.rules.mask <- colSums(subset.rules.mask, na.rm = TRUE) >= 2
  
  # Eliminar reglas redundates
  rule.set[!redundant.rules.mask]
}

# Cargar dataset
heart <- read.statlog.heart.transactions('./heart.dat')

  # Consultar información
summary(heart)
inspect(head(heart))

# Analizar el soporte de los ítems
items.support.relative <- sort(arules::itemFrequency(heart),
                               decreasing=TRUE)
items.support.absolute <- sort(arules::itemFrequency(heart,
                              type='absolute'), decreasing=TRUE)
items.support <- data.frame(soporte=items.support.relative,
                            n_apariciones=items.support.absolute,
                            row.names = names(items.support.relative))
items.support

# Representar gráficamente los soportes
pdf('./figuras/items_support.pdf')
graf <- ggplot2::ggplot(items.support,
                  aes(x=reorder(row.names(items.support), soporte),
                      y=soporte)) +
    ggplot2::xlab('ítems') +
    ggplot2::geom_bar(stat='identity', fill='coral3') +
    ggplot2::scale_y_continuous(breaks=scales::pretty_breaks(10)) +
    ggplot2::coord_flip()
graf
dev.off()

# Extracción de itemsets frecuentes  minSupport de 0.5
itemsets.freq1 <- arules::apriori(heart, parameter = list(support=0.5, target='frequent'))
itemsets.freq1 <- sort(itemsets.freq1, by='support')

# Extracción de itemsets frecuentes  minSupport de 0.3
itemsets.freq2 <- arules::apriori(heart, parameter = list(support=0.3, target='frequent'))
itemsets.freq2 <- sort(itemsets.freq2, by='support')
itemsets.freq2

summary(itemsets.freq2)

# Extracción de itemsets frecuentes  minSupport de 0.2
itemsets.freq3 <- arules::apriori(heart, parameter = list(support=0.2, target='frequent'))
itemsets.freq3 <- sort(itemsets.freq3, by='support')
itemsets.freq3

# Representar un diagrama de barras con el número de itemsets
# para cada número de ítems
length.itemsets.freq <- as.data.frame(table(size(itemsets.freq3)))
pdf('./figuras/longitud_itemsets.pdf')
graf <- ggplot2::ggplot(length.itemsets.freq, aes(x=Var1, y=Freq)) +
  ggplot2::geom_bar(colour='red', fill='white', stat='identity') +
  ggplot2::xlab('Longitud de ítems') + ggplot2::ylab('Itemsets')
graf
dev.off()

# Extraer reglas con un minConf de 0.9
rules.conf1 <- arules::apriori(heart, parameter = list(support=0.2,
                                                      confidence=0.9,
                                                      minlen=2))
rules.conf1 <- sort(rules.conf1, by='support')

#   Explorar las reglas extraídas
summary(rules.conf1)
inspect(head(rules.conf1))

#  Se eliminan las reglas redundantes
rules.conf1.pruned <- remove.redundat.rules(rules.conf1)
summary(rules.conf1.pruned)

  #   Se añade como métricas extra la confianza confirmada y la convicción:
extra.measures.rules.conf1 <- arules::interestMeasure(
  rules.conf1.pruned, measure=c('conviction', 'confirmedConfidence'),
  transactions = heart)

quality(rules.conf1.pruned) <-round(
      cbind(quality(rules.conf1.pruned), extra.measures.rules.conf1), 4)

# Extraer reglas con un minConf de 0.97 y un soporte máximo de 0.2
rules.conf2 <- arules::apriori(heart, parameter = list(confidence=0.97,
                                                       minlen=2,
                                                       smax=0.2))
summary(rules.conf2)

#  Se eliminan las reglas redundantes
rules.conf2.pruned <- remove.redundat.rules(rules.conf2)
summary(rules.conf2.pruned)


# Evaluar la convicción y Confianza confirmada en las reglas
extra.measures.rules.conf2 <- arules::interestMeasure(
  rules.conf2.pruned, measure=c('conviction', 'confirmedConfidence'),
  transactions = heart)

quality(rules.conf2.pruned) <-round(
  cbind(quality(rules.conf2.pruned), extra.measures.rules.conf2), 4)

# Explorar reglas de soporte mínimo 0.3037 y confianza mínima 0.6
rules.conf3 <- arules::apriori(heart, parameter = list(support=0.3037,
                                                       confidence=0.6,
                                                       minlen=2))

# Explorar las reglas extraídas
summary(rules.conf3)

#   Aplicar poda al conjunto de reglas rules.conf3
rules.conf3.pruned <- remove.redundat.rules(rules.conf3)

#   Calcular convicción y Confianza Confirmada sobre rules.conf3.prunned
extra.measures.rules.conf3 <- arules::interestMeasure(
  rules.conf3.pruned, measure=c('conviction', 'confirmedConfidence'),
  transactions = heart)

quality(rules.conf3.pruned) <-round(
  cbind(quality(rules.conf3.pruned), extra.measures.rules.conf3), 4)

# Nube de puntos del anterior conjunto de reglas
# Visualizar las reglas
pdf('./figuras/nube_reglas.pdf')
plot(rules.conf3.pruned, method = 'scatterplot',
     main = 'Nube de puntos de reglas',
     xlab = 'Soporte', ylab = 'Confianza')
dev.off()