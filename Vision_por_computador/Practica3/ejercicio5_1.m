%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: ejercicio5_1
% Pr�ctica 3 - Ejercicio 5.1
% Autor: Nicol�s Cubero
% Asignatura: Visi�n por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% 1. Cargar la imagen mujer.jpg y realiza las siguientes operaciones:

% Cargar la imagen mujer.jpg
img = imread('p3_imagenes/mujer.jpg');

% Visualizar la imagen original
figure,
subplot(1,2,1), imshow(img), title('Imagen original'),
subplot(1,2,2), imhist(img), title('Histograma');

%%

% Mejora el contraste usando �nicamente operaciones aritm�ticas (+,-,*,/)
manual_fix = 2*(img-100);

% Observando el histograma de la imagen, se observa que
% los colores de la imagen presentan una intensidad posterior a 100,
% por ello, se propone desplazar el histograma 100 unidades a la izquierda
% (oscurecer todos los colores restandoles una intensidad de 100) y
% multiplicar los valores de la imagen por 2 para extender la concentraci�n
% de los colores del histograma a lo largo de su dominio.

% Visualizar el arreglo manual
figure,
subplot(1,2,1), imshow(manual_fix), title('Ajuste manual de contraste'),
subplot(1,2,2), imhist(manual_fix), title('Histograma');

%%

% 2. Usa imadjust para mejorar el contraste. Usa la funci�n
% stretchlim

fun_fix = imadjust(img, stretchlim(img));

% Visualizar el arreglo con la funci�n imadjust
figure,
subplot(1,2,1), imshow(fun_fix), title('Ajuste con la funci�n imadjust'),
subplot(1,2,2), imhist(fun_fix), title('Histograma');

%%

% 3. Usa imadjust para aplicar una funci�n de transferencia de tipo gamma.
%  Comprueba el efecto que produce la transformaci�nen la imagen y en el
%  histograma

imgd = im2double(img);

hold on
k = 1;
for g=2.^[-3,-2,-1,1,2,3]
   % Aplicar transformaci�n gamma
    gamma_im = imadjust(imgd, [min(imgd(:)), max(imgd(:))], [0,1], g);
    
    subplot(6,2,k), imshow(gamma_im), title(sprintf('gamma %f', g));
    subplot(6,2,k+1), imhist(gamma_im), title('Histograma');
    
    k = k+2;
end
hold off

% An�lisis
%
% Para valores de gamma en [0,1], se observa que cuanto m�s se aproxima
% este par�metro a 0, el histograma tiende a desplazarse m�s a la derecha
% del histograma hacia los colores m�s claros y la amplitud del mismo
% tambi�n tiende a contraserse hacia este extremo. Al adoptar valores m�s
% cercanos a 1, el histograma tiende a extenderse m�s a lo largo de su
% dominio y a situar su centro de densidad en el cenrro del histograma. Por
% el contrario, al establecer valores de gamma con valores superiores a 1,
% el centro del histograma presenta una mayor tendencia a desplazarse hacia
% los colores m�s oscuros y a contraer su amplitud hacia este extremo.
%%

% Usa imadjust para aplicar la siguiente funci�n de transferencia:

tr_img = imgd; % Primer intervalo [0,100)
tr_img(tr_img<(100/255)) = 0;

tr_img = imadjust(imgd, [100/255, 1], [0,1]); % Segundo intervalo [100, 255]

% Mostrar imagen resultante

figure,
subplot(1,2,1), imshow(tr_img), title('Ajuste con funci�n de transferencia'),
subplot(1,2,2), imhist(tr_img), title('Histograma');

%%

% 4. Finalmente, ecualiza la imagen
img_eq = histeq(img);

% Mostrar imagen equalizada
figure,
subplot(1,2,1), imshow(img_eq), title('Imagen equalizada'),
subplot(1,2,2), imhist(img_eq), title('Histograma');
