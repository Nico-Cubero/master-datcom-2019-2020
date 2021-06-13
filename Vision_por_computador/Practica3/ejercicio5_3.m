%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_3
% Pr�ctica 3 - Ejercicio 5.3
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 3.Usa imadjust para aplicar la siguiente funci�n de transferencia a la
%   imagen campo.ppm

% Cargar la imagen
img = imread('p3_imagenes/campo.ppm');
imgd = im2double(img);

% Saturar a 0 todos los valores inferiores a 110
imgd(imgd<110/255) = 0;

% Saturar a 1 todos los valores superiores a 190
imgd(imgd>190/255) = 1;

% Aplicar transferencia en el intervalo (110, 190) con gamma 0.75
imgd = imadjust(imgd, [110, 190]/255, [0,1], 0.75);

% Visualizar la imagen
figure,
subplot(1,2,1), imshow(img), title('Imagen original'),
subplot(1,2,2), imshow(imgd), title('Imagen transformada');
