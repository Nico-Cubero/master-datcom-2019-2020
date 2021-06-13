%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio7_2
% Práctica 4 - Ejercicio 7.2
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar las imágenes
img_dist = imread('p4_imagenes/distorsion2.jpg');
img_ros = imread('p4_imagenes/rostro1.png');
img_ros2 = imread('p4_imagenes/rostro2.png');

% Mostrar las imágenes
figure,
subplot(1,3,1), imshow(img_dist), title('Distorsión'),
subplot(1,3,2), imshow(img_ros), title('Rostro 1'),
subplot(1,3,3), imshow(img_ros2), title('Rostro 2');

% Análisis
%
% Las imágenes Distorsión y Rostro 1, presentan algo de emborronamiento,
% por lo que convendría aplicar un filtro de paso alto para aumentar la
% nitidez de las imágenes, por su parte, la imagen Rostro 2 presenta algo
% de ruido blanco, por lo que en este caso, resultaría conveniente la
% aplicación de un filtro gaussiano.
%%
% Aplicación de las transformaciones

% Filtro nitidez
fNit = fspecial('unsharp');
%fLaplace = fspecial('laplacian', 0.25);

img_dist_nit = imfilter(img_dist, fNit, 'replicate'); %img_dist - imfilter(img_dist, fLaplace, 'replicate'); %fLaplace
img_ros_nit = imfilter(img_ros, fNit, 'replicate');

% Filtro gaussiano
fGauss = fspecial('gaussian', 5, 0.9);

img_ros2_filt = imfilter(img_ros2, fGauss);

% Representar las imágenes
figure,
subplot(2,3,1), imshow(img_dist), title('Distorsión Original'),
subplot(2,3,2), imshow(img_ros), title('Rostro 1 Original'),
subplot(2,3,3), imshow(img_ros2), title('Rostro 2 Original'),

subplot(2,3,4), imshow(img_dist_nit), title('Distorsión transformada'),
subplot(2,3,5), imshow(img_ros_nit), title('Rostro 1 transformada'),
subplot(2,3,6), imshow(img_ros2_filt), title('Rostro 2 transformada');
