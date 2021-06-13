%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio4_1
% Pr�ctica 1 - Ejercicio 4.1
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

I = imread('.\p1_imagenes\disney.png');

%%
% 1. Visualizarla con tres visualizadores

% Visualizaci�n con imshow
figure, imshow(I);

% Visualizaci�n con imtool
imtool(I);

% Visualizaci�n con imdesc
figure, imagesc(I);

% An�lisis
%
% - imshow() constituye la utilidad m�s b�sica de Matlab para visualizar una
%   imagen y permite usar otras funciones de zoom, desplazamiento de la imagen,
%   configuraci�n la visualizaci�n de la imagen (leyenda, t�tulo, etc),
%   as� como el almacenamiento de la misma, etc.
%
% - imtool() contituye otro visualizador que ofrece diversas funciones
%  para analizar m�s detalladamente los valores num�ricos asignados a cada
%  p�xel, recortar la imagen, modificar el contraste, etc.
%
% - imagesc() permite visualizar aplicando un reescalado din�mico de la
%    imagen y muestra los colores de la escala de grises con un mapa de
%    colores que facilita la distinci�n entre los p�xeles de color m�s
%    oscuro de los p�xeles de color m�s claro. Adem�s, a�ade escalas que
%    informan del ancho y alto de las im�genes.
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

% An�lisis de resultados
%
% Ambas funciones convierten el tipo de dato de las matrices con la imagen
% al tipo real double, sin embargo, im2double adem�s de realizar esta
% conversi�n, reescala los valores de la matriz de p�xeles al intervalo
% [0,1].
%
% Las funciones de visualizaci�n de im�genes, asumen que las im�genes de
% tipo double se hallan reescaladas a dicho intervalo, por lo que asumen
% que los valores iguales o pr�ximos a 0 se corresponden con p�xeles de
% color oscuro, mientras que los valores iguales o pr�ximos a 1 se
% corresponden con p�xeles de color claro. La funci�n double al no aplicar
% el reescalado de los valores lleva a obtener una imagen donde s�lo los
% p�xeles de color negro puro (valor 0) se colorean de color negro,
% mientras que el resto de p�xeles al tener un valor comprendido en [1,
% 255] son coloreados de color blanco. Puesto que la funci�n im2double s�
% aplica reescalado, se logra que los p�xeles se visualicen con los colores
% anteriores al reescalado.
