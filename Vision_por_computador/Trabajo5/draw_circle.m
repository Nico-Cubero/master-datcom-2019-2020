function img = draw_circle(u,v,r, sizeX, sizeY) 

    % Obtener el imagen con el círculo de radio r centrado en u y v
    [x, y] = meshgrid((1-sizeX/2):(sizeX/2), (1-sizeY/2):(sizeY/2));
    arg = (x-u).^2 + (y-v).^2;
    
    img = double(arg <= r^2);
end