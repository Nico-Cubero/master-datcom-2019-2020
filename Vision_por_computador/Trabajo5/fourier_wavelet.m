%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Archivo: fourier_wavelet.m
% Trabajo 5: Transformada de Fourier y Wavelent
% Autor: Nicolás Cubero
% Asignatura: Visión por Computador
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;

% Cargar la imagen cameraman
cam_img = imread('./cameraman.png');

% Pasar la imagen al dominio frecuencial y observar sus componentes
cam_ft = fft2(cam_img);             % Obtener la transformada de Fourier

% Crear el conjunto de discos a utilizar
clear disc_set;
disc_set{1} = struct('u', 0, 'v', 0, 'r', 1);
disc_set{2} = struct('u', 15, 'v', 0, 'r', 1);
disc_set{3} = struct('u', 12, 'v', 12, 'r', 1);
disc_set{4} = struct('u', 5, 'v', -4, 'r', 1);
disc_set{5} = struct('u', -5, 'v', 1, 'r', 1);
disc_set{6} = struct('u', -12, 'v', 10, 'r', 1);
disc_set{7} = struct('u', -3, 'v', -3, 'r', 1);

disc_set{8} = struct('u', 0, 'v', 0, 'r', 5);
disc_set{9} = struct('u', 15, 'v', 0, 'r', 5);
disc_set{10} = struct('u', 12, 'v', 12, 'r', 5);
disc_set{11} = struct('u', 5, 'v', -4, 'r', 5);
disc_set{12} = struct('u', -5, 'v', 1, 'r', 5);
disc_set{13} = struct('u', -12, 'v', 10, 'r', 5);
disc_set{14} = struct('u', -3, 'v', -3, 'r', 5);

disc_set{15} = struct('u', 0, 'v', 0, 'r', 15);
disc_set{16} = struct('u', 15, 'v', 0, 'r', 15);
disc_set{17} = struct('u', 12, 'v', 12, 'r', 15);
disc_set{18} = struct('u', 5, 'v', -4, 'r', 15);
disc_set{19} = struct('u', -5, 'v', 1, 'r', 15);
disc_set{20} = struct('u', -12, 'v', 10, 'r', 15);
disc_set{21} = struct('u', -3, 'v', -3, 'r', 15);

% Representar la componentes frecuenciales de cada disco
figure,
for i=1:numel(disc_set)
    
    % Obtener la imagen del círculo
    ft = draw_circle(disc_set{i}.u, disc_set{i}.v, disc_set{i}.r, size(cam_img,2), size(cam_img,1));
    
    ft_shift = fftshift(ft);
    
    % Obtener las frecuencias que pasan por el disco
    img_filter_ft = ft_shift.*cam_ft;

    % Pasar a dominio espacial
    img_filter = real(ifft2(img_filter_ft));
    
    % Representar componente real, imaginaria, módulo y fase
    subplot(3, ceil(numel(disc_set)/3), i),
    imshow(img_filter, []),
    title(sprintf('u=%d, v=%d, r=%d', disc_set{i}.u, disc_set{i}.v, disc_set{i}.r)),
end

%%

% Eliminación de ruido. Sobre la imagen cameraman insertar ruido gaussiano
% y mirar que componentes frecuenciales habría que eliminar para reducir el
% mayor ruido posible.

cam_img = imread('./cameraman.png');
cam_img = im2double(cam_img);

% Añadir ruido gaussiano a la imagen
cam_img = imnoise(cam_img,'gaussian',0, 0.1);

% Pasar la imagen al dominio frecuencial y observar sus componentes
cam_ft = fft2(cam_img);             % Obtener la transformada de Fourier
cam_ft_shift = fftshift(cam_ft);    % Centrar para determinar módulo y fase
    
p_real = real(cam_ft_shift);
p_img = imag(cam_ft_shift);
    
% Representar componente real, imaginaria, módulo y fase
figure,

subplot(2, 3, 1),
imshow(cam_img, []), title('Cameraman'),

subplot(2, 3, 2),
imshow(p_real, []), title('Componente real'),

subplot(2, 3, 3),
imshow(p_img, []), title('Componente imaginaria'),

subplot(2, 3, 4),
imshow(log(1+abs(cam_ft_shift)), []), title('Módulo (logaritmo)'),

subplot(2, 3, 5),
imshow(angle(cam_ft_shift), []), title('Fase');

%%%%% Aplicar un filtro paso bajo ideal

% Crear el filtro
low_pass_filter = double(zeros(size(cam_img, 1), size(cam_img, 2)));

x = (-size(low_pass_filter,1)/2+1):(size(low_pass_filter,1)/2);
y = (-size(low_pass_filter,2)/2+1):(size(low_pass_filter,2)/2);

[Y, X] = meshgrid(y, x);
dist = hypot(X,Y);

radius = 30;

ind = ind2sub(size(low_pass_filter), find(dist<=radius));
low_pass_filter(ind) = 1;
low_pass_filter_ft = fftshift(low_pass_filter);

% Aplicación del filtro sobre el dominio de Fourier
cam_ft_lpf = low_pass_filter_ft.*cam_ft;

% Pasar al dominio espacial la imagen convolucionada
cam_img_lpf = real(ifft2(cam_ft_lpf));

figure,
imshow(cam_img_lpf), title('Filtro paso bajo ideal');

%%%%% Aplicar el filtro de Butterworth

D=15; n=2;
but_filter = 1./(1+(dist./D).^(2*n));
but_filter_ft = fftshift(but_filter);

%Filtrado con butterworth
cam_ft_bf = but_filter_ft.*cam_ft;
cam_img_ft = real(ifft2(cam_ft_bf));

figure,
imshow(cam_img_ft), title('Filtro de Butterworth');

%%

% Realizar sobre la imagen barbara una descomposición wavelet usando
% bior3.7 con tres niveles. Fijado un porcentaje , por ejemplo 10 %, que 
% indica el porcentaje de coeficientes que nos quedamos de entre todos los
% coeficientes wavelets de la descomposición. Estos coeficientes son los
% que tiene mayor magnitud.
%
% Variar el procentaje y obtener una grafica en la que en el eje X tenemos
% razon de compresión y en el eje Y el valor de PSNR.

bar_img = imread('./barbara.png');

% Realizar descomposición wavelent con bior3.7 con 3 niveles
[C, S] = wavedec2(bar_img, 3, 'bior3.7');

% Ordenar las componentes según el valor de la magnitud sus coeficientes
[~, idx] = sort(abs(C), 'descend');

% Ratios de compresión considerados
rates = linspace(0.1, 1.0, 10);
psnr_errors = zeros(1, size(rates, 1));

figure,
for i=1:numel(rates)
    
    % Seleccionar las componentes de mayor magnitud
    highC = C;
    highC(idx(ceil(rates(i)*length(C)):length(C))) = 0;

    % Reconstrucción con ratio rates(i) de las componentes de mayor magnitud
    bar_rec = waverec2(highC, S, 'bior3.7');
    
    % Calcular el error PSNR cometido en esta aproximación
    [PSNR, ~, ~, ~] = measerr(bar_img, bar_rec);
    
    psnr_errors(i) = PSNR;
    
    % Representar la reconstrucción realizada
    subplot(2,length(rates)/2,i), imshow(bar_rec, []),
    title(sprintf('Reconstrucción %d %%', floor(rates(i)*100)));
end

% Reflejar ratio de compresión frente a error Peak Signal-to-noise ratio
figure,
plot(rates, psnr_errors, 'o-'),
title('Error PSNR cometido frente a ratios de compresión'),
xlabel('Ratio de compresión'),
ylabel('Error PSNR');
