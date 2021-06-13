function visualizeRGB(image)

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Comprobar parámetros          %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
   siz = size(image);
   
   if length(siz) ~= 3 || siz(3) ~=3
       error('La imagen especificada no es una imagen RGB válida');
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Procedimiento                 %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   % Representar la imagen junto con sus histogramas RGB
   figure,
   subplot(1,4,1), imshow(image), title('Imagen real'),
   subplot(1,4,2), imhist(image(:,:,1)), title('Banda roja'),
   subplot(1,4,3), imhist(image(:,:,2)), title('Banda verde'),
   subplot(1,4,4), imhist(image(:,:,3)), title('Banda azul');
end