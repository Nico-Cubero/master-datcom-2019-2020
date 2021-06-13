%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_5
% Pr�ctica 3 - Ejercicio 5.5
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 4.Aplica distintas t�cnicas de mejora de contraste sobre la imagen
% siguiente y compara los resultados

% Cargar la imagen
img = imread('p3_imagenes/paisaje.jpg');

% Convertir la imagen al espacio HSV
imgHSV = rgb2hsv(img);

% Representaci�n de los histogramas de cada banda RGB y cada banda HSV
figure,
subplot(2,3,1), imhist(img(:,:,1)), title('Histograma banda roja'),
subplot(2,3,2), imhist(img(:,:,2)), title('Histograma banda verde'),
subplot(2,3,3), imhist(img(:,:,3)), title('Histograma banda azul'),

subplot(2,3,4), imhist(imgHSV(:,:,1)), title('Histograma banda Hue'),
subplot(2,3,5), imhist(imgHSV(:,:,2)), title('Histograma banda Saturation'),
subplot(2,3,6), imhist(imgHSV(:,:,3)), title('Histograma banda Value');

%%

% Mejoras propuestas

% 1. Ajuste simple de cada banda RGB
imgCh = img;
imgCh(:,:,1) = imadjust(imgCh(:,:,1), stretchlim(imgCh(:,:,1)));
imgCh(:,:,2) = imadjust(imgCh(:,:,2), stretchlim(imgCh(:,:,2)));
imgCh(:,:,3) = imadjust(imgCh(:,:,3), stretchlim(imgCh(:,:,3)));

% 2. Ajuste simple del histograma de la banda Value del modelo HSV
imgAdjSimple = imgHSV;
imgAdjSimple(:,:,3) = imadjust(imgAdjSimple(:,:,3), stretchlim(imgAdjSimple(:,:,3)));
imgAdjSimple = hsv2rgb(imgAdjSimple);

% 3. Ajuste con correci�n gamma del histograma de la banda Value del modelo HSV
imgAdjGamma = imgHSV;
imgAdjGamma(:,:,3) = imadjust(imgAdjGamma(:,:,3), [min(min(imgAdjGamma(:,:,3))), max(max(imgAdjGamma(:,:,3)))], [0,1], 0.35);
imgAdjGamma= hsv2rgb(imgAdjGamma);

% 4. Ecualizaci�n simple de la banda Value del modelo HSV
imgEqSimple = imgHSV;
imgEqSimple(:,:,3) = histeq(imgEqSimple(:,:,3));
imgEqSimple = hsv2rgb(imgEqSimple);

% 5. Ecualizaci�n adaptativa de la banda Value del modelo HSV
imgEqAdapt = imgHSV;
imgEqAdapt(:,:,3) = adapthisteq(imgEqAdapt(:,:,3));
imgEqAdapt = hsv2rgb(imgEqAdapt);

% Visualizar los resultados
figure,
subplot(2,3,1),imshow(img), title('Imagen original'),
subplot(2,3,2),imshow(imgCh), title('Imagen con mejora en cada banda'),
subplot(2,3,3),imshow(imgAdjSimple), title('Ajuste simple'),
subplot(2,3,4),imshow(imgAdjGamma), title('Ajuste con correci�n gamma'),
subplot(2,3,5),imshow(imgEqSimple), title('Equalizaci�n simple'),
subplot(2,3,6),imshow(imgEqAdapt), title('Equalizaci�n adaptativa');

% An�lisis de resultados
%
% La trasnformaci�n que mejor permite disminuir los fuertes tonos oscuros
% presentes en la imagen original y que permite una visualizaci�n y
% distinci�n clara de los objetos presentes en la imagen es la
% transformaci�n con ajuste simple sobre la banda Value del espacio HSV con
% correci�n gamma.
%
% No obstante, esta transformaci�n a pesar de haber
% logrado una distinci�n y visualizaci�n clara de todos los elementos de la
% imagen, presenta el inconveniente de a�adir cierta artificialidad en los
% colores de la misma.
%
% Otras transformaciones que tambi�n permitir�an mejorar la visualizaci�n
% y distinci�n de los colores sin a�adir colores que resulten artificiales,
% son las transformaciones por ecualizaci�n simple y ecualizaci�n
% adaptativa ajustada al contraste sobre la banda Value del modelo de color
% HSV, sin embargo, estas transformaciones presentan el inconveniente de
% que siguen preservando los tonos oscuros fuertes de la imagen original en
% la imagen resultante.
