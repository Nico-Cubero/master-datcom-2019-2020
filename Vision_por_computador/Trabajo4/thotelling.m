clc; clear;

% Lectura de todas las im�genes
adra1 = imread('./banda1.tif');
adra2 = imread('./banda2.tif');
adra3 = imread('./banda3.tif');
adra4 = imread('./banda4.tif');
adra5 = imread('./banda5.tif');
adra6 = imread('./banda6.tif');

[n_rows, n_cols] = size(adra1);

% Transformar las im�genes al tipo double
adra1 = im2double(adra1);
adra2 = im2double(adra2);
adra3 = im2double(adra3);
adra4 = im2double(adra4);
adra5 = im2double(adra5);
adra6 = im2double(adra6);

% Colocar las im�genes como vectores verticales
adra1 = reshape(adra1', [], 1);
adra2 = reshape(adra2', [], 1);
adra3 = reshape(adra3', [], 1);
adra4 = reshape(adra4', [], 1);
adra5 = reshape(adra5', [], 1);
adra6 = reshape(adra6', [], 1);

% Colocar los vectores de im�genes en diferentes columnas
adra_imgs = cat(2, adra1, adra2, adra3, adra4, adra5, adra6);

% Calcular el vector media
mx = mean(adra_imgs(:));
%%

% Calcular la matriz de covarianzas entre las im�genes
C = cov(adra_imgs);

disp('Matriz de covarianza (C) de las im�genes:');
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

% Mostrar las im�genes resultantes de esta transformaci�n
figure
for i=1:size(trans, 2)
    subplot(2, size(trans, 2)/2, i),
    aux = reshape(trans(:,i), n_cols,  n_rows)';
    imshow(aux, []);
    title(sprintf('Transformada n�%d', i));
end

%%

% Reconstrucci�n de las im�genes transformadas

% Invertir la matriz de autovectores
A_inv = inv(A);

% Reconstruir las im�genes a partir de las transformaciones
x_rec = (A_inv*trans')' + mx;

% Mostrar las im�genes reconstruidas
figure,
for i=1:size(x_rec, 2)
    subplot(2, size(x_rec, 2)/2, i),
    aux = reshape(x_rec(:,i), n_cols,  n_rows)';
    imshow(aux);
    title(sprintf('Imagen reconstruida n�%d', i));
end

%%

% C�lculo del error cometido al aproximar la reconstrucci�n de las
% im�genes originales con las im�genes transformadas de Hotelling
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

% An�lisis
%
% En la anterior gr�fica, queda reflejado una reducci�n exponencial del
% error al incrementar el n�mero de transformadas usadas para aproximar
% la reconstrucci�n de las im�genes originales. De este modo, se observa
% que al considerar 2 transformadas se produce una dr�stica reducci�n del
% error y, a partir de la 3a transformada, el error de aproximaci�n
% es de un orden inferior a 10^-3.
%
% Por consiguiente, se puede realizar una reconstrucci�n muy aproximada
% de las im�genes originales contando �nicamente con 2 transformadas.