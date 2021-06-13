# M?ster -> Detecci?n de anomal?as
# Juan Carlos Cubero. Universidad de Granada
source('./!Outliers_A2_Librerias_a_cargar_en_cada_sesion.R')
###########################################################################
# UNIVARIATE STATISTICAL OUTLIERS -> IQR 
###########################################################################
source('!Outliers_A3_Funciones_a_cargar_en_cada_sesion.R')

###########################################################################
# RESUMEN:

# El objetivo es calcular los outliers 1-variantes, es decir, 
# con respecto a una ?nica variable o columna.
# Se va a utilizar el m?todo IQR que, aunque en principio, es s?lo aplicable
# a la distribuci?n normal, suele proporcionar resultados satisfactorios
# siempre que los datos no sigan distribuciones "raras" como por ejemplo 
# con varias modas o "picos"

# Da igual si los datos est?n normalizados o no.


############################################################################


# Cuando necesite lanzar una ventana gr?fica, ejecute X11()

# Vamos a trabajar con los siguientes objetos:

# mydata.numeric: frame de datos
# indice.columna: ?ndice de la columna de datos de mydata.numeric con la que se quiera trabajar
# nombre.mydata:  Nombre del frame para que aparezca en los plots

# En este script usaremos:

mydata.numeric  = mtcars[,-c(8:11)]  # mtcars[1:7]
indice.columna  = 1
nombre.mydata   = "mtcars"

# ------------------------------------------------------------------------

# Ahora creamos los siguientes objetos:

# mydata.numeric.scaled -> Debe contener los valores normalizados de mydata.numeric. Para ello, usad la funci?n scale
# columna -> Contendr? la columna de datos correspondiente a indice.columna. Basta realizar una selecci?n con corchetes de mydata.numeric
# nombre.columna -> Debe contener el nombre de la columna. Para ello, aplicamos la funci?n names sobre mydata.numeric
# columna.scaled -> Debe contener los valores normalizados de la anterior

mydata.numeric.scaled = scale(mydata.numeric)
columna         = mydata.numeric[, indice.columna]
nombre.columna  = names(mydata.numeric)[indice.columna]
columna.scaled  = mydata.numeric.scaled[, indice.columna]



###########################################################################
###########################################################################
# Parte primera. C?mputo de los outliers IQR
###########################################################################
###########################################################################



###########################################################################
# Calcular los outliers seg?n la regla IQR. Directamente sin funciones propias
###########################################################################

# Transparencia 80
quan <- quantile(columna.scaled)
iqr <- quan[4]-quan[2]

# ------------------------------------------------------------------------------------

# Calculamos las siguientes variables:

# cuartil.primero -> primer cuartil, 
# cuartil.tercero -> tercer cuartil
# iqr             -> distancia IQR

# Para ello, usamos las siguientes funciones:
# quantile(columna, x) para obtener los cuartiles
#    x=0.25 para el primer cuartil, 0.5 para la mediana y 0.75 para el tercero
# IQR para obtener la distancia intercuartil 
#    (o bien reste directamente el cuartil tercero y el primero)

# Calculamos las siguientes variables -los extremos que delimitan los outliers-

# extremo.superior.outlier.normal  = cuartil tercero + 1.5 IQR
# extremo.inferior.outlier.normal  = cuartil primero - 1.5 IQR
# extremo.superior.outlier.extremo = cuartil tercero + 3 IQR
# extremo.inferior.outlier.extremo = cuartil primero - 3 IQR
extremo.superior.outlier.normal <- quan[4] + 1.5*iqr
extremo.inferior.outlier.normal  = quan[2] - 1.5*iqr
extremo.superior.outlier.extremo = quan[4] + 3*iqr
extremo.inferior.outlier.extremo = quan[2] - 3*iqr

# Construimos sendos vectores: 

# vector.es.outlier.normal 
# vector.es.outlier.extremo

# Son vectores de valores l?gicos TRUE/FALSE que nos dicen 
# si cada registro es o no un outlier con respecto a la columna fijada
# Para ello, basta comparar con el operador > o el operador < la columna con alguno de los valores extremos anteriores

# El resultado debe ser el siguiente:
# [1] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
# [18] FALSE FALSE  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
vector.es.outlier.normal <- columna.scaled > extremo.superior.outlier.normal | columna.scaled < extremo.inferior.outlier.normal
vector.es.outlier.extremo <- columna.scaled > extremo.superior.outlier.extremo | columna.scaled < extremo.inferior.outlier.extremo





###########################################################################
# ?ndices y valores de los outliers
###########################################################################

# Construimos las siguientes variables:

# claves.outliers.normales     -> Vector con las claves (identificador num?rico de fila) de los valores que son outliers. Para obtenerlo, usad which sobre vector.es.outlier.normal
# data.frame.outliers.normales -> data frame obtenido con la selecci?n del data frame original de las filas que son outliers. Puede usarse o bien vector.es.outlier.normal o bien claves.outliers.normales
#                                 Este dataframe contiene los datos de todas las columnas de aquellas filas que son outliers.                                  
# nombres.outliers.normales    -> vector con los nombres de fila de los outliers. Para obtenerlo, usad row.names sobre el data frame anterior
# valores.outliers.normales    -> vector con los datos de los outliers. Se muestra s?lo el valor de la columna que se fij? al inicio del script 
# Idem con los extremos

claves.outliers.normales <- which(vector.es.outlier.normal)
data.frame.outliers.normales <- mydata.numeric[claves.outliers.normales,]
nombres.outliers.normales <- rownames(data.frame.outliers.normales)
valores.outliers.normales <- data.frame.outliers.normales[[nombre.columna]]

# Nos debe salir lo siguiente:

# claves.outliers.normales
# [1] 20

# data.frame.outliers.normales
#                  mpg cyl disp hp drat  wt qsec
# Toyota Corolla 33.9   4 71.1 65 4.22 1.835 19.9

# nombres.outliers.normales
# [1] "Toyota Corolla"

# valores.outliers.normales
# [1] 33.9

#  Outliers extremos no sale ninguno

claves.outliers.extremos <- which(vector.es.outlier.extremo)
data.frame.outliers.extremos <- mydata.numeric[claves.outliers.extremos,]
nombres.outliers.extremos <- rownames(data.frame.outliers.extremos)
valores.outliers.extremos <- data.frame.outliers.extremos[[nombre.columna]]







###########################################################################
# Desviaci?n de los outliers con respecto a la media de la columna
###########################################################################

# Construimos la variable:

# valores.normalizados.outliers.normales -> Contiene los valores normalizados de los outliers. 
# Usad columna.scaled y (o bien vector.es.outlier.normal o bien claves.outliers.normales)
desv <- data.frame.outliers.normales[[nombre.columna]] - mean(mydata.numeric[[nombre.columna]])
# Nos debe salir:
# valores.normalizados.outliers.normales
# Toyota Corolla   2.291272 


valores.normalizados.outliers.normales <- columna.scaled[claves.outliers.normales]




###########################################################################
# Plot
###########################################################################

# Mostramos en un plot los valores de los registros (los outliers se muestran en color rojo)
# Para ello, llamamos a la siguiente funci?n:
# MiPlot_Univariate_Outliers (columna de datos, indices -claves num?ricas- de outliers , nombre de columna)
# Lo hacemos con los outliers normales y con los extremos

#source('./!Outliers_A3_Funciones_a_cargar_en_cada_sesion.R')
MiPlot_Univariate_Outliers(columna,
                            claves.outliers.normales, nombre.columna)








###########################################################################
# BoxPlot
###########################################################################


# Vemos el diagrama de caja 

# Para ello, podr?amos usar la funci?n boxplot, pero ?sta no muestra el outlier en la columna mpg :-(
# Por lo tanto, vamos a usar otra funci?n. Esta es la funci?n geom_boxplot definida en el paquete ggplot
# En vez de usarla directamente, llamamos a la siguiente funci?n:
# MiBoxPlot_IQR_Univariate_Outliers = function (datos, indice.de.columna, coef = 1.5)
# Esta funci?n est? definida en el fichero de funciones A3 que ha de cargar previamente.
# Esta funci?n llama internamente a geom_boxplot

# Una vez que la hemos llamado con mydata.numeric y con indice.columna, la volvemos
# a llamar pero con los datos normalizados.
# Lo hacemos para resaltar que el Boxplot es el mismo ya que la normalizaci?n
# no afecta a la posici?n relativa de los datos 
MiBoxPlot_IQR_Univariate_Outliers(mydata.numeric, indice.columna)
MiBoxPlot_IQR_Univariate_Outliers(mydata.numeric.scaled, indice.columna)






###########################################################################
# C?mputo de los outliers IQR con funciones propias
###########################################################################

# En este apartado hacemos lo mismo que antes, pero llamando a funciones que est?n dentro de !Outliers_A3_Funciones.R :

# vector_es_outlier_IQR = function (datos, indice.de.columna, coef = 1.5)  
# datos es un data frame y coef es el factor multiplicativo en el criterio de outlier,
# es decir, 1.5 por defecto para los outliers normales y un valor mayor para outliers extremos (3 usualmente)
# -> devuelve un vector TRUE/FALSE indicando si cada dato es o no un outlier


# vector_claves_outliers_IQR = function(datos, indice, coef = 1.5)
# Funci?n similar a la anterior salvo que devuelve los ?ndices de los outliers



ourliers.mask <- vector_es_outlier_IQR(datos, indice.columna,)




###########################################################################
# BoxPlot
###########################################################################


# Mostramos los boxplots en un mismo gr?fico.
# Tenemos que usar los datos normalizados, para que as? sean comparables


# Llamamos a la funci?n  MiBoxPlot_Juntos
# MiBoxPlot_juntos  = function (datos, vector_TF_datos_a_incluir)  
# Pasamos mydata.numeric como par?metro a datos.
# Si no pasamos nada como segundo par?metro, se incluir?n todos los datos en el c?mputo.
# Esta funci?n normaliza los datos y muestra, de forma conjunta, los diagramas de cajas
# As?, podemos apreciar qu? rango de valores toma cada outlier en las distintas columnas.

# Para etiquetar los outliers en el gr?fico
# llamamos a la funci?n MiBoxPlot_juntos_con_etiquetas 
MiBoxPlot_juntos(datos, outliers.mask)















