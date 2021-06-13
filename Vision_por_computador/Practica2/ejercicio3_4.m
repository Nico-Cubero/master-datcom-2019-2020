%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio3_4
% Práctica 2 - Ejercicio 3.4
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

F=imread('imagenes_chromakey/chromakey_original.jpg');
B=imread('imagenes_chromakey/praga1.jpg');

result = chroma_key2(F,B,size(B,2)-size(F,2), size(B,1)-size(F,1), 0, 255, 0);

figure,
imshow(result),
title('Composición generada');
