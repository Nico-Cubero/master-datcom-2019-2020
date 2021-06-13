# Librerías importados
library('arules')

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
  dat[,13] <- factor(dat[,13], levels=c(3,6,7), labels=c(
                                              'Normal',
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
    dat[['maximum heart rate achieved']],
    method = 'frequency', breaks=4)
  
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
  redundant.rules.mask <- colSums(subset.rules.mask,
                                  na.rm = TRUE) >= 2
  
  # Eliminar reglas redundates
  rule.set[!redundant.rules.mask]
}

# Cargar dataset
heart <- read.statlog.heart.transactions('./heart.dat')

# Extraer reglas con un minConf de 0.9
rules.conf1 <- arules::apriori(heart, parameter = list(support=0.2,
                                                  confidence=0.9,
                                                  minlen=2))
rules.conf1 <- sort(rules.conf1, by='support')

#  Se eliminan las reglas redundantes
rules.conf1.pruned <- remove.redundat.rules(rules.conf1)

# Extraer reglas con un minConf de 0.97 y un soporte máximo de 0.2
rules.conf2 <- arules::apriori(heart, parameter = list(confidence=0.97,
                                                   minlen=2,
                                                   smax=0.2))

#  Se eliminan las reglas redundantes
rules.conf2.pruned <- remove.redundat.rules(rules.conf2)

# Explorar reglas de soporte mínimo 0.3037 y confianza mínima 0.6
rules.conf3 <- arules::apriori(heart, parameter = list(support=0.3037,
                                                   confidence=0.6,
                                                   minlen=2))

#   Aplicar poda al conjunto de reglas rules.conf3
rules.conf3.pruned <- remove.redundat.rules(rules.conf3)

#   Seleccionar sólo las reglas que se eligieron
rules.conf3.pruned <- subset(rules.conf3.pruned,
                             subset=support>0.336 & confidence>0.7537)

# Combinar todas las reglas en un sólo conjunto
rule.set <- union(union(rules.conf1.pruned, rules.conf2.pruned),
                  rules.conf3.pruned)

# Calcular métricas extra
extra.measures.rules.set <- arules::interestMeasure(rule.set,
                                          measure=c('confirmedConfidence',
                                                    'lift'),
                                          transactions = heart)
quality(rule.set) <-round(
  cbind(quality(rule.set), extra.measures.rules.set),
  4)

# Analizar las reglas con heart dissease=FALSE
rules.heart_dissease.false <- subset(rule.set,
        subset=rhs %in% 'heart dissease=FALSE')

# Analizar las reglas con heart dissease=TRUE
rules.heart_dissease.true <- subset(rule.set,
        subset=rhs %in% 'heart dissease=TRUE')

# Analizar las reglas con sex=male y heart dissease=TRUE en el antecedente o consecuente
rules.male_heart_dissease <- subset(rule.set,
            subset=lhs %in% c('heart dissease=TRUE',
                              'heart dissease=FALSE',
                              'sex=male', 'sex=female') &
              rhs %in% c('heart dissease=TRUE', 
                         'heart dissease=FALSE',
                         'sex=male', 'sex=female'))

# Búsqueda de otros grupos de reglas que no llevó al descubrimiento de ningún hecho relevante
inspect(subset(rule.set,
     subset=lhs %in% c('exercise induced angina=FALSE',
                                 'exercise induced angina=TRUE') |
       rhs %in% c('exercise induced angina=FALSE',
                  'exercise induced angina=TRUE')))

inspect(subset(rule.set,
     subset=lhs %in% c('number of major vessels=0',
                                 'number of major vessels=1',
                                 'number of major vessels=2',
                                 'number of major vessels=3') |
       rhs %in% c('number of major vessels=0',
                  'number of major vessels=1',
                  'number of major vessels=2',
                  'number of major vessels=3')))

inspect(subset(rule.set,
       subset=lhs %in% c('thal=Normal',
                         'thal=Reversable defect',
                         'thal=Fixed defect',
                         'heart dissease=TRUE',
                         'heart dissease=FALSE') &
         rhs %in% c('thal=Normal', 'thal=Reversable defect',
                    'thal=Fixed defect', 'heart dissease=TRUE',
                    'heart dissease=FALSE')))

inspect(subset(rule.set,
       subset=lhs %in% c('serum cholestoral=Dangerous level',
                         'serum cholestoral=High level',
                         'serum cholestoral=Normal level') |
         rhs %in% c('serum cholestoral=Dangerous level',
                    'serum cholestoral=High level',
                    'serum cholestoral=Normal level')))

inspect(subset(rule.set, subset=lhs %in% c(
  'resting electrocardiographic results=left ventricular hypertrophy',
  'resting electrocardiographic results=Normal',
  'resting electrocardiographic results=ST-T wave anormality') |
                 rhs %in% c(
  'resting electrocardiographic results=left ventricular hypertrophy',
  'resting electrocardiographic results=Normal',
  'resting electrocardiographic results=ST-T wave anormality')))

inspect(subset(rule.set, subset=lhs %in% c(
  'slope of the peak exercise ST segment=Upsloping',
  'slope of the peak exercise ST segment=Downsloping',
  'slope of the peak exercise ST segment=Flat') |
                 rhs %in% c(
  'slope of the peak exercise ST segment=Upsloping',
  'slope of the peak exercise ST segment=Downsloping',
  'slope of the peak exercise ST segment=Flat')))

inspect(subset(rule.set, subset=lhs %in% c(
  'chest pain type=asymptomatic',
  'chest pain type=non-anginal pain',
  'chest pain type=atypical angina',
  'chest pain type=typical angina') |
                 rhs %in% c(
  'chest pain type=asymptomatic',
  'chest pain type=non-anginal pain',
  'chest pain type=atypical angina',
  'chest pain type=typical angina')))

inspect(subset(rule.set, subset=lhs %in% c(
  'oldpeak=[0,0.1)',
  'oldpeak=[0.1,1.4)',
  'oldpeak=[1.4,6.2]') |
                 rhs %in% c(
  'oldpeak=[0,0.1)',
  'oldpeak=[0.1,1.4)', 'oldpeak=[1.4,6.2]')))

inspect(subset(rule.set,
   subset=lhs %in% c('maximum heart rate achieved=[71,133)',
                     'maximum heart rate achieved=[133,154)',
                     'maximum heart rate achieved=[166,202]') |
     rhs %in% c('maximum heart rate achieved=[71,133)',
                'maximum heart rate achieved=[133,154)',
                'maximum heart rate achieved=[166,202]')))