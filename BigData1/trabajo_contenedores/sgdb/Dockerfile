FROM mysql:latest
LABEL maintainer Nicolás Cubero

# Habilitar el modo de autenticación al nativo con contraseña
CMD ["mysqld", "--default-authentication-plugin=mysql_native_password"]

# Establecer la contraseña de roor
ENV MYSQL_ROOT_PASSWORD "tarara_blanca82"

# Crear base de datos
ENV MYSQL_DATABASE "Gestion_trabajadores"

# Crear usuario datcom y establecer su contraseña
ENV MYSQL_USER "administrador"
ENV MYSQL_PASSWORD "granada_lorca32"

# Iniciar el servidor de MySQL
EXPOSE 3306
