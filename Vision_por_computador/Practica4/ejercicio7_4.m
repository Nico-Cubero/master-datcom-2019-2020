%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio7_4
% Práctica 4 - Ejercicio 7.4
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar la imagen distorsion
img = imread('p4_imagenes/distorsion1.jpg');

% Aplicar filtro Gaussiano
fgaus = fspecial('gaussian',7, 1.25); % Filtro de tamaño 5x5 y std 0.6
img_gaus = imfilter(img, fgaus, 'replicate');

% Aplicar filtro motion
fmotion = fspecial('motion',7, 180); 
img_motion = imfilter(img, fmotion, 'replicate');

% Visualizar la imagen
figure,
subplot(1,3,1), imshow(img), title('Imagen original'),
subplot(1,3,2), imshow(img_gaus), title('Imagen con filtro gausiano'),
subplot(1,3,3), imshow(img_motion), title('Imagen con filtro de movimiento');

% Análisis
%
% La imagen de entrada presenta un tipo de ruido introducido por el propio
% dipositivo de captura. Aplicamos un filtro gaussiano, así como un filtro
% de movimiento para tratar de disminuirlo, de modo que a simple vista, se
% observa que el filtro que añade movimiento logra disminuir en mayor
% medida la aparición de este ruido al tiempo que incrementa de forma
% más notable el emborronamiento de la imagen.
%
% Otra de las técnicas que se podría aplicar para obtener una mejora sobre
% la calidad de la imagen es una ecualización que permitiría mejorar el
% contraste de la imagen

% Aplicar ecualización sobre la imagen con el filtro motion
img_medianHSV = rgb2hsv(img_motion);
img_medianHSV(:,:,3) = adapthisteq(img_medianHSV(:,:,3));
img_medianHSV = hsv2rgb(img_medianHSV);

figure,
subplot(1,2,1), imshow(img), title('Imagen original'),
subplot(1,2,2), imshow(img_medianHSV), title('Imagen con filtro de movimiento y ecualización');
