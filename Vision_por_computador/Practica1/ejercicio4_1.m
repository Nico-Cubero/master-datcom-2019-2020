%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio4_1
% Práctica 1 - Ejercicio 4.1
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = imread('.\p1_imagenes\disney.png');

%%
% 1. Visualizarla con tres visualizadores

% Visualización con imshow
figure, imshow(I);

% Visualización con imtool
imtool(I);

% Visualización con imdesc
figure, imagesc(I);

% Análisis
%
% - imshow() constituye la utilidad más básica de Matlab para visualizar una
%   imagen y permite usar otras funciones de zoom, desplazamiento de la imagen,
%   configuración la visualización de la imagen (leyenda, título, etc),
%   así como el almacenamiento de la misma, etc.
%
% - imtool() contituye otro visualizador que ofrece diversas funciones
%  para analizar más detalladamente los valores numéricos asignados a cada
%  píxel, recortar la imagen, modificar el contraste, etc.
%
% - imagesc() permite visualizar aplicando un reescalado dinámico de la
%    imagen y muestra los colores de la escala de grises con un mapa de
%    colores que facilita la distinción entre los píxeles de color más
%    oscuro de los píxeles de color más claro. Además, añade escalas que
%    informan del ancho y alto de las imágenes.
%%
% 2. Convertir a tipo double con double(img) y visualizar de nuevo.

I_d = double(I);

% Visualizar esta nueva imagen obtenida
figure, imshow(I_d);

%%
% 3. Convertir a double con im2double y analizar el resultado.
I_d2 = im2double(I);

% Visualizar esta imagen junto con la original y la obtenida en el paso 2
figure,
sibplot(1,3,1), imshow(I), title('Imagen original')
subplot(1,3,2), imshow(I_d), title('Imagen de valores reales sin reescalado'),
subplot(1,3,3), imshow(I_d2), title('Imagen de valores reales reescalados');

% Análisis de resultados
%
% Ambas funciones convierten el tipo de dato de las matrices con la imagen
% al tipo real double, sin embargo, im2double además de realizar esta
% conversión, reescala los valores de la matriz de píxeles al intervalo
% [0,1].
%
% Las funciones de visualización de imágenes, asumen que las imágenes de
% tipo double se hallan reescaladas a dicho intervalo, por lo que asumen
% que los valores iguales o próximos a 0 se corresponden con píxeles de
% color oscuro, mientras que los valores iguales o próximos a 1 se
% corresponden con píxeles de color claro. La función double al no aplicar
% el reescalado de los valores lleva a obtener una imagen donde sólo los
% píxeles de color negro puro (valor 0) se colorean de color negro,
% mientras que el resto de píxeles al tener un valor comprendido en [1,
% 255] son coloreados de color blanco. Puesto que la función im2double sí
% aplica reescalado, se logra que los píxeles se visualicen con los colores
% anteriores al reescalado.
