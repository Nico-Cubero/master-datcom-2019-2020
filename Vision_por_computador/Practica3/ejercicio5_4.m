%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_4
% Práctica 3 - Ejercicio 5.4
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 4. Usa la función adapthisteq sobre la imagen mujer.jpg y analiza
% el resultado

% Cargar la imagen
img = imread('p3_imagenes/mujer.jpg');

% Aplicar ecualización simple del histograma
img_eq = histeq(img);

% Aplicar ecualización adaptavivo limitada por el contraste
img_fix = adapthisteq(img);

% Visualizar la imagen resultante
figure,
subplot(3,2,1), imshow(img), title('Imagen original'),
subplot(3,2,2), imhist(img),
subplot(3,2,3), imshow(img_eq), title('Ecualización simple'),
subplot(3,2,4), imhist(img_eq),
subplot(3,2,5), imshow(img_fix), title('Ecualización adaptativa'),
subplot(3,2,6), imhist(img_fix);

% Análisis
%
% La ecualización adaptativa limitada por el contraste permite llevar a
% cabo la mejora del contraste más adecuada para cada región de la imagen,
% de modo que el histograma resultante no resulta tan dirupto como el que
% se obtiene en la ecualización simple.
% homogénea el contraste 
