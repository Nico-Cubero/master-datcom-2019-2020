function img = draw_circle_freq(u,v,r, sizeX, sizeY) 

    % Crear imagen vacía compleja
    img = zeros(sizeX, sizeY);
    img = complex(img, 0);

    % Obtener el imagen con el círculo de radio r centrado en u y v
    [x, y] = meshgrid(1:sizeX, 1:sizeY);
    mask = (x-u).^2 + (y-v).^2 <= r^2;
    
    img(mask) = 1;
    
    %img = double(img);
end