%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio7_5
% Práctica 4 - Ejercicio 7.5
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 5. Obtener sobre la imagen formas.png las esquinas usando el método de
% Harris.

% Cargar imagen de formas
img_formas = im2double(imread('p4_imagenes/formas.png'));

% Aplicar el filtro de Harris
corners = detectHarrisFeatures(img_formas);

% Representar ls imagen con las esquinas detectadas
imshow(img_formas);
hold on
plot(corners.Location(:,1), corners.Location(:,2), 'r+');
title('Formas con bordes detectados');
hold off
