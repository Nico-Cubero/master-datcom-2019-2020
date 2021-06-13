%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio4_2
% Pr�ctica 1 - Ejercicio 4.2
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar imagen
img = imread('./p1_imagenes/rosa.jpg');

%% 

% 1. Visualizar las tres componentes de manera simult�nea junto con la
%   imagen original yanalizar los resultados

% Separar los 3 canales
r_img = img(:,:,1);
g_img = img(:,:,2);
b_img = img(:,:,3);

% Representar la imagen con sus 3 canales
figure,
subplot(1,4,1), imshow(img), title('Color');
subplot(1,4,2), imshow(r_img), title('Banda roja'),
subplot(1,4,3), imshow(g_img), title('Banda verde'),
subplot(1,4,4), imshow(b_img), title('Banda azul');

% An�lisis
%
% En la anterior figura se puede visualizar a la izquierda la imagen
% original en color, las 3 im�genes a la derecha muestran de forma separada
% cada una de las bandas rojo, verde y azul que conjuntamente definen el
% color de cada p�xel de la imagen original. En estas im�genes, los p�xeles
% con color m�s pr�ximo al blanco se corresponden con aquellos p�xeles de
% la imagen original donde existe mayor presencia del color rojo, verde y/o
% azul, mientras que los p�xeles pr�ximos al negro se corresponden con los
% p�xeles de la imagen original donde existe ausencia de color rojo, verde
% o azul.
%% 

% 2. Anula una de sus bandas (por ejemplo la roja) y analiza los
% resultados. Se recomienda usar imtool para ver los valores de color en
% cada p�xel

% Copiamos la imagen en otra sin la banda roja
img_mod = img;
img_mod(:,:,1) = 0;

% Representamos la imagen
figure, imshow(img_mod), title('Imagen sin banda roja');

% An�lisis
%
% Al anular la banda roja, se obtiene una imagen sin informaci�n de color
% rojo y, por lo tanto, carente de tonos rojos.

%%

% 3. Usa tambi�n la imagen sintetica.jpg, haz otras modifiaciones y observa
% los resultados (por ejemplo: poner una de sus bandas al nivel m�ximo,
% intercambiar el papel de las bandas entre s�, aplicar un desplazamiento a
% alguna de las bandas con circshift, invertir alguna de sus bandas con
% flplr o flpud, ...).

sint_img = imread('./p1_imagenes/sintetica.jpg');

% Poner la banda verde al m�ximo
sint_img_g_max = sint_img;
sint_img_g_max(:,:,2) = 255;

% Visualizar resultados
figure,
subplot(1,2,1), imshow(sint_img), title('Imagen original'),
subplot(1,2,2), imshow(sint_img_g_max), title('Banda verde color m�ximo');

% An�lisis
%
% Al establecer los valores de la banda verde a su valor m�ximo, se
% a�ade a la imagen una tonalidad verde total a todos los p�xeles.

% Intercambiar la banda roja y la azul
sint_img_change = sint_img;

aux = sint_img_change(:,:,1);
sint_img_change(:,:,1) = sint_img_change(:,:,3);
sint_img_change(:,:,3) = aux;

% Mostrar la imagen resultante
figure,
subplot(1,2,1), imshow(sint_img), title('Imagen original'),
subplot(1,2,2), imshow(sint_img_change), title('Intercambio de banda roja y azul');

% An�lisis
%
% En este caso, se ha obtenido una imagen con colores diferentes a la
% imagen original, donde los p�xeles con mayor coloraci�n roja en la
% imagen original, ahora tienen coloraci�n azul en la misma intensidad y
% los p�xeles con mayor coloraci�n azul en la imagen original, pasan a
% tener coloraci�n roja en la misma intensidad.
%
% Debido a que en la imagen original, en este ejemplo, no hab�a ning�n
% elemento en color azul, en la imagen transformada tampoco se distingue
% ning�n elemento en color rojo.
