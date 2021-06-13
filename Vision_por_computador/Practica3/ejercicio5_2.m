%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_2
% Pr�ctica 3 - Ejercicio 5.2
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar la imagen danza
img = imread('p3_imagenes/danza.ppm');

% Equalizaci�n de cada banda RGB
imgRGB = img;

imgRGB(:,:,1) = histeq(imgRGB(:,:,1));
imgRGB(:,:,2) = histeq(imgRGB(:,:,2));
imgRGB(:,:,3) = histeq(imgRGB(:,:,3));

% Equalizaci�n de cada banda hsv
imgHSV = rgb2hsv(img);

imgHSV(:,:,1) = histeq(imgHSV(:,:,1));
imgHSV(:,:,2) = histeq(imgHSV(:,:,2));
imgHSV(:,:,3) = histeq(imgHSV(:,:,3));

imgHSV = hsv2rgb(imgHSV);

% Visualizaci�n de la imagen resultante
figure,
subplot(1,3,1), imshow(img), title('Imagen original'),
subplot(1,3,2), imshow(imgRGB), title('Ecualizaci�n de cada banda RGB'),
subplot(1,3,3), imshow(imgHSV), title('Ecualizaci�n de cada banda HSV');

% An�lisis
%
% La ecualizaci�n de la imagen con las 2 anteriores estrategias produce
% im�genes con colores poco realistas e incluso descoloreadas.
%
% Para lograr una equalizaci�n m�s adecuada, conviene transformar la imagen
% al espacio HSV y aplicar equalizaci�n sobre el canal Value ya que este
% recoge informaci�n sobre la luminosidad de los colores de cada p�xel de
% la imagen, dado que mediante la operaci�n de ecualizaci�n se pretende
% extender la concentraci�n de intensidad de la imagen a lo largo de todos
% sus colores, resulta m�s adecuado extender el histograma de la banda Value
% para lograr este extensi�n de intensidad en todos los colores de la
% imagen.

% Equalizaci�n del canal V de HSV
imgHSV2 = rgb2hsv(img);
imgHSV2(:,:,3) = histeq(imgHSV2(:,:,3));

img_eq = hsv2rgb(imgHSV2);

% Mostrar la imagen
figure,
subplot(1,2,1), imshow(img), title('Imagen original'),
subplot(1,2,2), imshow(img_eq), title('Ecualizaci�n');
