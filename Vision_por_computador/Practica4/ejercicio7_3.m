%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio7_3
% Práctica 4 - Ejercicio 7.3
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 3. Utiliza la correlación para buscar formas en una imagen.

% Cargar las imágenes que se van a emplear en la búsqueda
img_formas = im2double(imread('p4_imagenes/formas.png'));
img_texto = rgb2gray(im2double(imread('p4_imagenes/texto.png')));

% Cargar los patrones a detectar
estrella = im2double(imread('p4_imagenes/estrella.png'));
ovalo = im2double(imread('p4_imagenes/ovalo.png'));
cuadrado = im2double(imread('p4_imagenes/cuadrado.png'));
cuadrado2 = im2double(imread('p4_imagenes/cuadrado2.png'));
cuadrado3 = im2double(imread('p4_imagenes/cuadrado3.png'));

clear formas;
formas{1} = estrella;
formas{2} = ovalo;
formas{3} = cuadrado;
formas{4} = cuadrado2;
formas{5} = cuadrado3;



letra_i = im2double(imread('p4_imagenes/letra_i.png'));
letra_k = im2double(imread('p4_imagenes/letra_k.png'));
letra_m = im2double(imread('p4_imagenes/letra_m.png'));
letra_o = im2double(imread('p4_imagenes/letra_o.png'));
letra_p = im2double(imread('p4_imagenes/letra_p.png'));

clear letras;

letras{1} = letra_i;
letras{2} = letra_k;
letras{3} = letra_m;
letras{4} = letra_o;
letras{5} = letra_p;

% Detección sobre la imagen formas
formas_det = zeros(size(img_formas));

for i=1:numel(formas)
    aux = imfilter(img_formas, formas{i});
    
    % Reescalar valores al intervalo [0,1] y poner en blanco los valores
    % iguales a 1
    aux = (aux - min(aux(:))) / (max(aux(:))-min(aux(:)));
    aux = aux > (1 - 1e-7); % Tolerancia de error eps = 10^-7
    
    formas_det = formas_det + aux;
end

% Detección sobre la imagen letras
letras_det = zeros(size(img_texto));

img_texto_inv = 1-img_texto; % Poner las letras en blanco y el fondo negro

for i=1:numel(letras)
    aux = imfilter(img_texto_inv, 1-letras{i});
    
    % Reescalar valores al intervalo [0,1] y poner en blanco los valores
    % iguales a 1
    aux = (aux - min(aux(:))) / (max(aux(:))-min(aux(:)));
    aux = aux > (1 - 1e-7); % Tolerancia de error eps = 10^-7
    
    letras_det = letras_det + aux;
end

% Visualización de la imagen original y la detección
figure,
subplot(2,2,1), imshow(img_formas), title('Imagen Formas'),
subplot(2,2,2), imshow(formas_det), title('Formas detectadas'),

subplot(2,2,3), imshow(img_texto), title('Imagen texto'),
subplot(2,2,4), imshow(letras_det), title('Letras detectadas');
