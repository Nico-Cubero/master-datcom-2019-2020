FROM python:3.8
LABEL maintainer Nicolás Cubero

# Instalar Open SSH
RUN apt-get -y update
RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Crear usuario "administrador" y asignar contraseña
RUN useradd -s /bin/bash administrador
RUN echo "administrador:granada_lorca32" | chpasswd

RUN pip install numpy # Instalar numpy
RUN pip install pandas # Instalar pandas
RUN pip install scipy # Instalar scipy
RUN pip install scikit-learn # Instalar scikit-learn
RUN pip install matplotlib # Instalar matplotlib

# Copiar el script de ejemplo al contenedor
ADD ejemplo_svm_iris.py /

# Habilitar el demonio ssh y el puerto
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
