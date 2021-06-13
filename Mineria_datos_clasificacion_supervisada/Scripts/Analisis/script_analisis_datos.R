#########################################################################
# Nombre: script_analisis_datos.R
# Descripción: Script interactivo empleado para analizar y visualizar la
#             distribución de los datos del dataset
# Uso: Este script debe de ser ejecutado de forma interactiva, ejecutando
#       de forma manual cada una de las sentencias que lo componen
# Autor: Nicolás Cubero
#########################################################################

library('ggplot2')
library('dplyr')

dat <- read.csv('../train_values_4910797b-ee55-40a7-8668-10efd5c1b960.csv',
                sep=',', header=T, na.strings = c('','-'))


# Comprobar si alguna fila tiene NA
any(is.na(dat))

# Analizar amount_tsh
ggplot2::ggplot(dat, aes(amount_tsh)) +
  ggplot2::geom_histogram(binwidth = 5000, colour='black', fill='white') +
  scale_y_log10()

# Diagrama de puntos de longitud y latitud
ggplot2::ggplot(dat, aes(y=latitude, x=longitude)) +
  geom_point(alpha=0.3, colour='red', size=0.2)

# Diagrama de puntos de longitud frente a gps_height
ggplot2::ggplot(dat %>% filter(longitude>0 & latitude<0),
                aes(x=longitude, y=latitude, colour=gps_height)) +
  geom_point(alpha=0.3, size=0.2)

# Analizar gps_height
summary(dat$gps_height)

#   Diagrama de cajas
ggplot2::ggplot(dat, aes(x='gps_height', y=gps_height)) +
  ggplot2::geom_boxplot(outlier.alpha = 0.8, outlier.colour = 'red',
                        outlier.shape = 8)

# Analizar histograma de gps_height
ggplot2::ggplot(dat, aes(gps_height)) +
  ggplot2::geom_histogram(binwidth = 70, colour='black', fill='white')

# Convertir date_recorded en el tipo Date
dat$date_recorded <- as.Date(dat$date_recorded)
summary(dat$date_recorded)

#   Dibujamos un diagrama de cajas
ggplot2::ggplot(dat, aes(y=date_recorded)) +
  ggplot2::geom_boxplot(outlier.alpha =0.4 , outlier.colour = 'red',
                        outlier.shape = 0.8)

#   Y la dibujamos
ggplot2::ggplot(dat, aes(date_recorded)) +
  scale_y_log10() +
  geom_histogram(binwidth = 30, colour = 'red', fill = 'white')

# Convertir construction_year a año
summary(dat$construction_year)

#   Y la dibujamos
ggplot2::ggplot(dat, aes(construction_year)) +
  geom_histogram(binwidth = 5, colour = 'red', fill = 'white') +
  scale_y_log10()

# Analizar los valores mayores que 0
dat.construction_year = dat %>% select(construction_year) %>%
  filter(construction_year>0)
summary(dat.construction_year)
#   Y la dibujamos
ggplot2::ggplot(dat.construction_year, aes(x=construction_year)) +
  geom_histogram(binwidth = 5, colour = 'red', fill = 'white') +
  scale_y_log10()

# Analizar payment y payment_type
str(dat$payment)
str(dat$payment_type)

summary(dat$payment)
summary(dat$payment_type)

# Dibujar payment
ggplot2::ggplot(dat, aes(payment)) +
  ggplot2::geom_bar(colour='red', fill='white') +
  theme(text=element_text(size=13))

# Dibujar payment_type
ggplot2::ggplot(dat, aes(payment_type)) +
  ggplot2::geom_bar(colour='red', fill='white') +
  theme(text=element_text(size=13))

# Comprobar relación entre funder e installer
str(dat$funder)
str(dat$installer)

summary(dat$funder)
summary(dat$installer)

# Dibujar installer
ggplot2::ggplot(dat, aes(x=installer)) +
  ggplot2::geom_bar() +
  coord_flip()

#Tabla de contingencia de funder e installer
funder_installer <- table(dat$funder, dat$installer)

# Analizar num_private
summary(dat$num_private)

#   Diagrama de barras
ggplot2::ggplot(dat, aes(num_private)) +
  ggplot2::geom_bar(fill='white', colour='grey') +
  scale_y_log10() +
  scale_x_log10()

# Analizar basin
basin.table <- table(dat$basin)
prop.table(basin.table)

# Analizar region_code
summary(dat$region_code)

#   Diagrama de barras
ggplot2::ggplot(dat, aes(region_code)) +
  ggplot2::geom_bar(fill='white', colour='red')

# Analizar district_code
summary(dat$district_code)

ggplot2::ggplot(dat, aes(district_code)) +
  ggplot2::geom_bar(fill='white', colour='red')

# Analziar population
summary(dat$population)

#   Visualizar boxplot
ggplot2::ggplot(dat, aes(x='population', y=population)) +
  ggplot2::geom_boxplot(outlier.alpha =0.4 , outlier.colour = 'red',
                        outlier.shape = 0.8)

#   Diagrama de barras
ggplot2::ggplot(dat, aes(population)) +
  ggplot2::geom_bar(fill='white', colour='red') +
  scale_x_log10()

# Analizar public_meeting
summary(dat$public_meeting)

ggplot2::ggplot(dat, aes(public_meeting)) +
  ggplot2::geom_bar(fill='white', colour='red')

# Analizar scheme_management
summary(dat$scheme_management)

ggplot2::ggplot(dat, aes(scheme_management)) +
  ggplot2::geom_bar(fill='white', colour='red') +
  ggplot2::coord_flip()

# Analizar permit
summary(dat$permit)

ggplot2::ggplot(dat, aes(permit)) +
  ggplot2::geom_bar(fill='white', colour='red')

# Analizar management
ggplot2::ggplot(dat, aes(management)) +
  ggplot2::geom_bar(fill='white', colour='red') +
  coord_flip()

# Analizar management_group
ggplot2::ggplot(dat, aes(management_group, fill=management)) +
  ggplot2::geom_bar( colour='black')

# Analizar water_quality
summary(dat$water_quality)

# Analizar water_quality
ggplot2::ggplot(dat, aes(water_quality)) +
  ggplot2::geom_bar(fill='white', colour='red')

# Analizar quality_group
ggplot2::ggplot(dat, aes(quality_group, fill=water_quality)) +
  ggplot2::geom_bar( colour='red')
