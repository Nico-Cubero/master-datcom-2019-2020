%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_4
% Pr�ctica 3 - Ejercicio 5.4
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 4. Usa la funci�n adapthisteq sobre la imagen mujer.jpg y analiza
% el resultado

% Cargar la imagen
img = imread('p3_imagenes/mujer.jpg');

% Aplicar ecualizaci�n simple del histograma
img_eq = histeq(img);

% Aplicar ecualizaci�n adaptavivo limitada por el contraste
img_fix = adapthisteq(img);

% Visualizar la imagen resultante
figure,
subplot(3,2,1), imshow(img), title('Imagen original'),
subplot(3,2,2), imhist(img),
subplot(3,2,3), imshow(img_eq), title('Ecualizaci�n simple'),
subplot(3,2,4), imhist(img_eq),
subplot(3,2,5), imshow(img_fix), title('Ecualizaci�n adaptativa'),
subplot(3,2,6), imhist(img_fix);

% An�lisis
%
% La ecualizaci�n adaptativa limitada por el contraste permite llevar a
% cabo la mejora del contraste m�s adecuada para cada regi�n de la imagen,
% de modo que el histograma resultante no resulta tan dirupto como el que
% se obtiene en la ecualizaci�n simple.
% homog�nea el contraste 
