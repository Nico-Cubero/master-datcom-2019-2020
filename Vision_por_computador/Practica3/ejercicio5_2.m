%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_2
% Práctica 3 - Ejercicio 5.2
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar la imagen danza
img = imread('p3_imagenes/danza.ppm');

% Equalización de cada banda RGB
imgRGB = img;

imgRGB(:,:,1) = histeq(imgRGB(:,:,1));
imgRGB(:,:,2) = histeq(imgRGB(:,:,2));
imgRGB(:,:,3) = histeq(imgRGB(:,:,3));

% Equalización de cada banda hsv
imgHSV = rgb2hsv(img);

imgHSV(:,:,1) = histeq(imgHSV(:,:,1));
imgHSV(:,:,2) = histeq(imgHSV(:,:,2));
imgHSV(:,:,3) = histeq(imgHSV(:,:,3));

imgHSV = hsv2rgb(imgHSV);

% Visualización de la imagen resultante
figure,
subplot(1,3,1), imshow(img), title('Imagen original'),
subplot(1,3,2), imshow(imgRGB), title('Ecualización de cada banda RGB'),
subplot(1,3,3), imshow(imgHSV), title('Ecualización de cada banda HSV');

% Análisis
%
% La ecualización de la imagen con las 2 anteriores estrategias produce
% imágenes con colores poco realistas e incluso descoloreadas.
%
% Para lograr una equalización más adecuada, conviene transformar la imagen
% al espacio HSV y aplicar equalización sobre el canal Value ya que este
% recoge información sobre la luminosidad de los colores de cada píxel de
% la imagen, dado que mediante la operación de ecualización se pretende
% extender la concentración de intensidad de la imagen a lo largo de todos
% sus colores, resulta más adecuado extender el histograma de la banda Value
% para lograr este extensión de intensidad en todos los colores de la
% imagen.

% Equalización del canal V de HSV
imgHSV2 = rgb2hsv(img);
imgHSV2(:,:,3) = histeq(imgHSV2(:,:,3));

img_eq = hsv2rgb(imgHSV2);

% Mostrar la imagen
figure,
subplot(1,2,1), imshow(img), title('Imagen original'),
subplot(1,2,2), imshow(img_eq), title('Ecualización');
