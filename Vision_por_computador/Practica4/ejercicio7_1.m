%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio7_1
% Práctica 4 - Ejercicio 7.1
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar las imágenes
img1 = imread('p4_imagenes/disney_r1.png');
img2 = imread('p4_imagenes/disney_r2.png');
img3 = imread('p4_imagenes/disney_r3.png');
img4 = imread('p4_imagenes/disney_r4.png');
img5 = imread('p4_imagenes/disney_r5.png');

% Aplicación de filtros gaussianos
fgaus = fspecial('gaussian',5, 0.6); % Filtro de tamaño 5x5 y std 0.6

img1_gaus = imfilter(img1, fgaus, 'replicate');
img2_gaus = imfilter(img2, fgaus, 'replicate');
img3_gaus = imfilter(img3, fgaus, 'replicate');
img4_gaus = imfilter(img4, fgaus, 'replicate');
img5_gaus = imfilter(img5, fgaus, 'replicate');

% Aplicación de filtros de mediana
img1_med = medfilt2(img1, [5, 5]);
img2_med = medfilt2(img2, [5, 5]);
img3_med = medfilt2(img3, [5, 5]);
img4_med = medfilt2(img4, [5, 5]);
img5_med = medfilt2(img5, [5, 5]);

% Mostrar las imágenes resultantes
figure,
subplot(1,3,1), imshow(img1), title('Disney 1'),
subplot(1,3,2), imshow(img1_gaus), title('Filtro Gausiano 5x5, std=0.6'),
subplot(1,3,3), imshow(img1_med), title('Filtro mediana 5x5');

figure,
subplot(1,3,1), imshow(img2), title('Disney 2'),
subplot(1,3,2), imshow(img2_gaus), title('Filtro Gausiano 5x5, std=0.6'),
subplot(1,3,3), imshow(img2_med), title('Filtro mediana 5x5');

figure,
subplot(1,3,1), imshow(img3), title('Disney 3'),
subplot(1,3,2), imshow(img3_gaus), title('Filtro Gausiano 5x5, std=0.6'),
subplot(1,3,3), imshow(img3_med), title('Filtro mediana 5x5');

figure,
subplot(1,3,1), imshow(img4), title('Disney 4'),
subplot(1,3,2), imshow(img4_gaus), title('Filtro Gausiano 5x5, std=0.6'),
subplot(1,3,3), imshow(img4_med), title('Filtro mediana 5x5');

figure,
subplot(1,3,1), imshow(img5), title('Disney 5'),
subplot(1,3,2), imshow(img5_gaus), title('Filtro Gausiano 5x5, std=0.6'),
subplot(1,3,3), imshow(img5_med), title('Filtro mediana 5x5');

% Análisis
%
% Las imágenes tratadas en este fichero, presentan un tipo de ruido
% conocido como ruido "sal y pimienta", consistente en la aparición de
% colores extremos en píxeles aleatorios de la imagen.
%
% De este modo, se puede comprobar que el filtro mediana permite una mejor
% eliminación de este ruido sal y pimienta, obteniendo excelentes
% resultados salvo en las últimas imágenes en las que la presencia de ruido
% es mayor. El filtro gaussiano al basarse en un primedio ponderado según
% una distribución gaussiana, resulta más sensible a los valores extremos
% que presentados por este tipo de ruido, mientras que el filtro mediana,
% resulta más robusto a este tipo de ruido, por lo que logra llevar a cabo
% un mejor filtrado.
