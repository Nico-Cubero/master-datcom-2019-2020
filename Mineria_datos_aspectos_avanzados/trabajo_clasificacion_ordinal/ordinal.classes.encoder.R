ordinal.classes.encoder <- function(X, class.col, labels.ordered=NULL) {

  # Función para la transformación de la variable
  # de clase categórica de un dataset en variables
  # de clase binarias ordinales dada una relación
  # de orden entre los valores de clase indicada
  #
  # Recibe:
  # X: Dataset con una variable de clase categórico
  #     debe de pasarse como dataframe con nombres
  #     de columna no nulos.
  # class.col: Nombre o número de la columna del dataframe
  #           que expresa la clase.
  # labels.orderd: Valores de la clase expresados
  #               según el orden en el que deben de
  #               ordenados de izquierda a derecha.
  #               Si no se especifica, se tratará
  #               de obtener de la propia columna.
  #
  # Devuelve: Dataframe con las variables independientes
  #           del dataset y las variables de clase binarias
  #           ordinales.
    
  # Check parameters input
  if (!(is.data.frame(X)) || is.matrix(X)) {
    stop('X must be a dataframe or a matrix')
  } else if (is.null(colnames(X))) {
    colnames(X) <- paste0('C',1:ncol(X))
  }
  
  if (length(class.col)!=1) {
    stop('Only one value allowed for "label.col"')
  }
  
  # Número de columna de clase
  class.column.num <- NA
  
  # Tomar el número de la columna de clase
  if (is.character(class.col)) {
    
    if (!class.col %in% colnames(X)) {
      stop(paste('"',class.col,'" not included in dataset'))
    }
    
    class.column.num <- which(colnames(X)==class.col)
    
  } else if (is.integer(class.col)) {
    
    if (class.col < 1 || class.col>ncol(X)) {
      stop('Column ',class.col,' not a valid column')
    }
    
    class.column.num <- class.col
    
  } else {
    stop('Only character or integers allowed for "class.col"')
  }
  
  # Las etiquetas pasadas deben de ser valores de etiqueta de la clase
  if (is.null(labels.ordered)) {
    
    # Tomar orden de los valores de la columna de clase si ningún orden es especificado
    if (is.ordered(X[,class.column.num]) || is.numeric(X[,class.column.num])) {
      labels.ordered <- sort(unique(X[,class.column.num]))
    }
    else {
      stop('Cannot deduce implicit order from class')
    }
    
  } else if (any(!labels.ordered %in% unique(X[,class.column.num]))) {
    warning('Any label in "labels.ordered" not a class of X')
  }
  
  # Nuevo conjunto de etiquetas de clase con información ordinal
  target <- matrix(data=rep(NA, nrow(X)*(length(labels.ordered)-1)),
                   nrow=nrow(X))
  
  # Para todas las etiquetas de la clase
  for (i in 1:(ncol(target))) {
    
    # Los índices para la etiqueta de clase i 
    index.class <- X[,class.column.num]==labels.ordered[i]
    
    #Copiar las etiquetas de la anterior columna ordinal que representa
    # una clase de orden inferior
    if (i>1) {
      target[target[,i-1]==0,i] <- 0
    }
    
    # Generar columna con 0's para la etiqueta i y 1's para el resto
    target[index.class,i] <- 0
    target[is.na(target[,i]),i] <- 1
  }
  
  # Convertir target en columnas de dataframe y agregarlo al df original
  target <- as.data.frame(target)
  target[] <- lapply(target, factor, ordered=TRUE)
  
  colnames(target) <- paste(colnames(X)[class.column.num],'_', 1:ncol(target), sep='')
  
  # Reemplazar la columna de clase original por las ordinales
  X_new <- cbind(X[,-class.column.num], target)
  
  return(X_new)
}
