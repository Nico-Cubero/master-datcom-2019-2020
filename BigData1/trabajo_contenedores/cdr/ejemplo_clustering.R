# Paquetes importados
library('dplyr')

set.seed(7)

# Usar dataset iris y separar los datos de la etiqueta usando dplyr
# (se podría haber usado el operador de slicing, pero se prefiere testear tidyverse)
attach(iris)
iris.data <- iris %>% select(c("Sepal.Length", "Sepal.Width", "Petal.Length",
                              "Petal.Width"))
iris.class <- iris %>% select('Species')

# Normalizar dataset
iris.mean <- apply(iris.data, MARGIN = 2, FUN=mean)
iris.std <- apply(iris.data, MARGIN = 2, FUN=sd)

iris.data <- t(apply(iris.data, MARGIN=1,
              FUN=function(X,mean,std){(X-mean)/std},
              iris.mean, iris.std))

# Ejecutar clustering
clust <- kmeans(iris.data, centers=3)

# Mostrar los resultados
cat('Número de instancias agrupadas en cada cluster:', fill=TRUE)
print(table(clust$cluster))

cat('Distancia SS asociada al clustering: ', clust$totss, fill=T)

cat('Distancia SS intercluster: ',
    sum(dist(clust$centers, method = 'manhattan')^2), fill=T)
