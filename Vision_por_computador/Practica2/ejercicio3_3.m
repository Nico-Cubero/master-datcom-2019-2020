%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio3_3
% Pr�ctica 2 - Ejercicio 3.3
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar imagen de ejemplo
img = imread('./p2_imagenes/Warhol_Marilyn_1967_OnBlueGround.jpg');

% Convertir la imagen al espacio HSV y modificar la componente Hue
img_hsv = rgb2hsv(img);
hue_offset = 60/255; % Constante que se a�ade a la componente hue

img_hsv(:,:,1) = img_hsv(:,:,1) + hue_offset;
img = hsv2rgb(img_hsv); % Volver a convertir a RGB

% Visualizar la imagen junto a los histogramas de cada banda HSV
visualizeHSV(img);

% An�lisis
%
% Al a�adir la constante hue_offset a la banda Matiz del espacio de color
% HSV en la imagen, se provoca un desplazamiento en la coloraci�n de la
% imagen, de modo que la imagen resultante presenta un conjunto de colores
% diferentes a los de la imagen original.
