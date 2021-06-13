%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio4_2
% Práctica 1 - Ejercicio 4.2
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar imagen
img = imread('./p1_imagenes/rosa.jpg');

%% 

% 1. Visualizar las tres componentes de manera simultánea junto con la
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

% Análisis
%
% En la anterior figura se puede visualizar a la izquierda la imagen
% original en color, las 3 imágenes a la derecha muestran de forma separada
% cada una de las bandas rojo, verde y azul que conjuntamente definen el
% color de cada píxel de la imagen original. En estas imágenes, los píxeles
% con color más próximo al blanco se corresponden con aquellos píxeles de
% la imagen original donde existe mayor presencia del color rojo, verde y/o
% azul, mientras que los píxeles próximos al negro se corresponden con los
% píxeles de la imagen original donde existe ausencia de color rojo, verde
% o azul.
%% 

% 2. Anula una de sus bandas (por ejemplo la roja) y analiza los
% resultados. Se recomienda usar imtool para ver los valores de color en
% cada píxel

% Copiamos la imagen en otra sin la banda roja
img_mod = img;
img_mod(:,:,1) = 0;

% Representamos la imagen
figure, imshow(img_mod), title('Imagen sin banda roja');

% Análisis
%
% Al anular la banda roja, se obtiene una imagen sin información de color
% rojo y, por lo tanto, carente de tonos rojos.

%%

% 3. Usa también la imagen sintetica.jpg, haz otras modifiaciones y observa
% los resultados (por ejemplo: poner una de sus bandas al nivel máximo,
% intercambiar el papel de las bandas entre sí, aplicar un desplazamiento a
% alguna de las bandas con circshift, invertir alguna de sus bandas con
% flplr o flpud, ...).

sint_img = imread('./p1_imagenes/sintetica.jpg');

% Poner la banda verde al máximo
sint_img_g_max = sint_img;
sint_img_g_max(:,:,2) = 255;

% Visualizar resultados
figure,
subplot(1,2,1), imshow(sint_img), title('Imagen original'),
subplot(1,2,2), imshow(sint_img_g_max), title('Banda verde color máximo');

% Análisis
%
% Al establecer los valores de la banda verde a su valor máximo, se
% añade a la imagen una tonalidad verde total a todos los píxeles.

% Intercambiar la banda roja y la azul
sint_img_change = sint_img;

aux = sint_img_change(:,:,1);
sint_img_change(:,:,1) = sint_img_change(:,:,3);
sint_img_change(:,:,3) = aux;

% Mostrar la imagen resultante
figure,
subplot(1,2,1), imshow(sint_img), title('Imagen original'),
subplot(1,2,2), imshow(sint_img_change), title('Intercambio de banda roja y azul');

% Análisis
%
% En este caso, se ha obtenido una imagen con colores diferentes a la
% imagen original, donde los píxeles con mayor coloración roja en la
% imagen original, ahora tienen coloración azul en la misma intensidad y
% los píxeles con mayor coloración azul en la imagen original, pasan a
% tener coloración roja en la misma intensidad.
%
% Debido a que en la imagen original, en este ejemplo, no había ningún
% elemento en color azul, en la imagen transformada tampoco se distingue
% ningún elemento en color rojo.
