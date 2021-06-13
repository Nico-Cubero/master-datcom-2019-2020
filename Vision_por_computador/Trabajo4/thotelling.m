clc; clear;

% Lectura de todas las imágenes
adra1 = imread('./banda1.tif');
adra2 = imread('./banda2.tif');
adra3 = imread('./banda3.tif');
adra4 = imread('./banda4.tif');
adra5 = imread('./banda5.tif');
adra6 = imread('./banda6.tif');

[n_rows, n_cols] = size(adra1);

% Transformar las imágenes al tipo double
adra1 = im2double(adra1);
adra2 = im2double(adra2);
adra3 = im2double(adra3);
adra4 = im2double(adra4);
adra5 = im2double(adra5);
adra6 = im2double(adra6);

% Colocar las imágenes como vectores verticales
adra1 = reshape(adra1', [], 1);
adra2 = reshape(adra2', [], 1);
adra3 = reshape(adra3', [], 1);
adra4 = reshape(adra4', [], 1);
adra5 = reshape(adra5', [], 1);
adra6 = reshape(adra6', [], 1);

% Colocar los vectores de imágenes en diferentes columnas
adra_imgs = cat(2, adra1, adra2, adra3, adra4, adra5, adra6);

% Calcular el vector media
mx = mean(adra_imgs(:));
%%

% Calcular la matriz de covarianzas entre las imágenes
C = cov(adra_imgs);

disp('Matriz de covarianza (C) de las imágenes:');
disp(C);

% Calcular autovectores y autovalores de la matriz de covarianzas
[A, D] = eig(C, 'vector');

% Ordenarlos en orden decreciente de autovalor
[D, ind] = sort(D, 'descend');
A = A(:, ind)';

D = D';

disp('Autovectores de la matriz de covarianzas (A):');
disp(A);

disp('Autovalores de la matriz de covarianzas (D):');
disp(D);

%%

% Calculamos la transformada de Hoetling
trans = (A*(adra_imgs - mx)')';

% Mostrar las imágenes resultantes de esta transformación
figure
for i=1:size(trans, 2)
    subplot(2, size(trans, 2)/2, i),
    aux = reshape(trans(:,i), n_cols,  n_rows)';
    imshow(aux, []);
    title(sprintf('Transformada nº%d', i));
end

%%

% Reconstrucción de las imágenes transformadas

% Invertir la matriz de autovectores
A_inv = inv(A);

% Reconstruir las imágenes a partir de las transformaciones
x_rec = (A_inv*trans')' + mx;

% Mostrar las imágenes reconstruidas
figure,
for i=1:size(x_rec, 2)
    subplot(2, size(x_rec, 2)/2, i),
    aux = reshape(x_rec(:,i), n_cols,  n_rows)';
    imshow(aux);
    title(sprintf('Imagen reconstruida nº%d', i));
end

%%

% Cálculo del error cometido al aproximar la reconstrucción de las
% imágenes originales con las imágenes transformadas de Hotelling
error = zeros(1, size(trans, 2));

for i=1:size(error, 2)-1
    error(i) = sum(D(i+1:end));
end

% Mostar los errores por pantalla
for i=1:size(error, 2)
    error(i) = sum(D(i+1:end));
    fprintf('Error cometido al aproximar con los %d primeras transformadas: %f\n', i, error(i));
end

figure,
plot(error, 'o-'),
title('Error por transformadas'),
xlabel('Transformadas usadas')
ylabel('Error MSE cometido')

% Análisis
%
% En la anterior gráfica, queda reflejado una reducción exponencial del
% error al incrementar el número de transformadas usadas para aproximar
% la reconstrucción de las imágenes originales. De este modo, se observa
% que al considerar 2 transformadas se produce una drástica reducción del
% error y, a partir de la 3a transformada, el error de aproximación
% es de un orden inferior a 10^-3.
%
% Por consiguiente, se puede realizar una reconstrucción muy aproximada
% de las imágenes originales contando únicamente con 2 transformadas.