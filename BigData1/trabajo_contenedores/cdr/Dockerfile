FROM rocker/r-base
LABEL maintainer Nicolás Cubero

# Actualizar e instalar todos los paquetes de sistema
RUN apt-get -y update
RUN apt-get -y install libcurl4-openssl-dev xml2 libxml2-dev libssl-dev

# Instalar Open SSH
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Crear usuario "administrador" y asignar contraseña
RUN useradd -s /bin/bash administrador
RUN echo "administrador:granada_lorca32" | chpasswd

# Instalar todos los paquetes de R
RUN R -e "install.packages(c('caret', 'RSNNS', 'frbs', 'FSinR', 'forecast', 'e1071', 'tidyverse'))"

# Añadir el script de ejemplo
ADD ejemplo_clustering.R /

# Habilitar el demonio ssh y el puerto
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
