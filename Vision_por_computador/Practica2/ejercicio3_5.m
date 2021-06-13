%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio3_5
% Práctica 2 - Ejercicio 3.5
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar imagen de ejemplo
img = imread('./p2_imagenes/Warhol_Marilyn_1967_OnBlueGround.jpg');

% Preparar el clip de vídeo
clear peli;

% Convertir imagen al espacio de colores hsv
img_hsv = rgb2hsv(img);

inc = 0.01;
n_frames = ceil(1/inc);

for i=1:n_frames
    % Editar banda hue y añadirla como nuevo fotograma del clip
    img_hsv(:,:,1) = mod(img_hsv(:,:,1)+ inc, 1.0);
    peli(i)=im2frame((hsv2rgb(img_hsv)));
end

% Mostrar clip de vídeo
figure,
movie(peli);
