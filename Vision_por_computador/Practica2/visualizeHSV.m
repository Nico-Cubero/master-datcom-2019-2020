function visualizeHSV(image)

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Comprobar parámetros          %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   %if ismatrix(image) == 0
   %    error('La imagen proporcionada no es una imagen válida');
   %end
   
   siz = size(image);
   
   if length(siz) ~= 3 || siz(3) ~=3
       error('La imagen especificada no es una imagen RGB válida');
   end

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Procedimiento                 %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   % Convertir al imagen al espacio HSV
   img_hsv = rgb2hsv(image);
   c = hsv(256);

   % Representar la imagen junto con sus histogramas HSV
   figure,
   subplot(1,4,1), imshow(image), title('Imagen real'),
   subplot(1,4,2), imhist(img_hsv(:,:,1)*255, c), title('Matiz'),
   subplot(1,4,3), imhist(img_hsv(:,:,2)), title('Saturación'),
   subplot(1,4,4), imhist(img_hsv(:,:,3)), title('Matiz');
end