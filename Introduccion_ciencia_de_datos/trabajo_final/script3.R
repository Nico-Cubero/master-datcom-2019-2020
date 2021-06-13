###############################################################################
# Nombre del script: script3.R
# Desarrollado por: Nicolás Cubero
# Descripción: Script para el desarrollo de tests estadísticos para realizar
# la comparación de modelos
# Nota: Este script ha sido desarrollado para ejecutarse de forma interactiva
#       línea por línea
###############################################################################
# Script para la comparación de modelos
results.train <- read.csv('./regr_train_alumnos.csv', row.names = 1)
results.test <- read.csv('./regr_test_alumnos.csv', row.names = 1)

# Sobre el conjunto de train
# Comparar out_train_lm con out_train_kknn (referencia) con Wilcoxon
difs <- (results.train[,1] - results.train[,2]) / results.train[,1]
wilc_1_2 <- cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                  ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) <- c(colnames(results.train)[1], colnames(results.train)[2])
head(wilc_1_2)

# Aplicar test y calcular R+ y R-
LMvsKNNtst <- wilcox.test(wilc_1_2[,1], wilc_1_2[,2], alternative = "two.sided", paired=TRUE)
Rmas <- LMvsKNNtst$statistic
pvalue <- LMvsKNNtst$p.value

LMvsKNNtst <- wilcox.test(wilc_1_2[,2], wilc_1_2[,1], alternative = "two.sided", paired=TRUE)
Rmenos <- LMvsKNNtst$statistic

cat('Test modelo lineal (R+) vs modelo k-NN:(R-)', fill=T)
cat('Valor R+: ',Rmas, fill=T)
cat('Valor R-: ',Rmenos, fill=T)
cat('p-value del test: ',pvalue, fill=T)

# Sobre el conjunto de test
# Comparar out_train_lm con out_train_kknn (referencia) con Wilcoxon
difs <- (results.test[,1] - results.test[,2]) / results.test[,1]
wilc_1_2 <- cbind(ifelse (difs<0, abs(difs)+0.1, 0+0.1),
                  ifelse (difs>0, abs(difs)+0.1, 0+0.1))
colnames(wilc_1_2) <- c(colnames(results.test)[1], colnames(results.test)[2])
head(wilc_1_2)

# Aplicar test y calcular R+ y R-
LMvsKNNtst <- wilcox.test(wilc_1_2[,1], wilc_1_2[,2], alternative = "two.sided", paired=TRUE)
Rmas <- LMvsKNNtst$statistic
pvalue <- LMvsKNNtst$p.value

LMvsKNNtst <- wilcox.test(wilc_1_2[,2], wilc_1_2[,1], alternative = "two.sided", paired=TRUE)
Rmenos <- LMvsKNNtst$statistic

cat('Test modelo lineal (R+) vs modelo k-NN:(R-)', fill=T)
cat('Valor R+: ',Rmas, fill=T)
cat('Valor R-: ',Rmenos, fill=T)
cat('p-value del test: ',pvalue, fill=T)

# Aplicar el test de Friedman sobre el conjunto de entrenamiento
test_friedman <- friedman.test(as.matrix(results.train))
test_friedman

# Aplicar el test post-hoc de Holm para averiguar qué par es diferente
tam <- dim(results.train)
groups <- rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(results.train), groups, p.adjust = "holm", paired = TRUE)

# Aplicar el test de Friedman sobre el conjunto de test
test_friedman <- friedman.test(as.matrix(results.test))
test_friedman

# Aplicar el test post-hoc de Holm para averiguar qué par es diferente
tam <- dim(results.test)
groups <- rep(1:tam[2], each=tam[1])
pairwise.wilcox.test(as.matrix(results.train), groups, p.adjust = "holm", paired = TRUE)
