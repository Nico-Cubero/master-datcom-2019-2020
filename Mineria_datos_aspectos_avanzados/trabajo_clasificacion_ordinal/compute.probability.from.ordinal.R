compute.probability.from.ordinal <- function(P) {
  
  # Función que calcula la probabilidad de clases
  # a partir de un conjunto de probabilidades ordinales
  # dadas por un conjunto de clasificadores binarios
  # ordinales.
  #
  # Recibe: P: Array de (n_instancias x 2 x prob ordinales)
  # Devuelve: Matriz de nº instancias x nº de clases con las
  #           probabilidades de pertenencia de cada clase.
  
  # Calcular número de clases y el de instancias
  n_class <- dim(P)[3] + 1
  n_inst <- dim(P)[1]
  
  # Matriz final con la probabilidades de cada instancia
  # para cada clase
  prob_class <- matrix(data = 1, ncol = n_class, nrow = n_inst)
  
  # Matriz auxiliar para calcular probabilidades condicionales
  aux_cond_prob <- matrix(data = 1, ncol = 1, nrow = n_inst)
  
  # La probabilidad de la clase 1 ya se conoce
  prob_class[,1] <- P[,1,1]
  
  for(i in 2:(n_class-1)) {
    
    # Añadir y calcular P(C>Ci-1)
    aux_cond_prob = aux_cond_prob*P[,2,i-1]
    
    # Calcular P(C=Ci)
    prob_class[,i] <- aux_cond_prob*P[,1,i]
  }
  
  # Asignar la probabilidad de la úlima clase
  prob_class[,ncol(prob_class)] <- P[,2,n_class-1]*aux_cond_prob
  
  return(prob_class)
}