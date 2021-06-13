function result = chroma_key2(F, B, col, row, Rcolor, Gcolor, Bcolor, varargin )

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Comprobar par�metros          %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   sizF = size(F);
   
   if length(sizF) ~= 3 || sizF(3) ~=3
       error('La imagen especificada no es una imagen RGB v�lida');
   end
   
   sizB = size(B);
   
   if length(sizB) ~= 3 || sizB(3) ~=3
       error('El fondo especificado no es una imagen RGB v�lida');
   end
   
   if ceil(col) ~= col
       error('La columna de inicio para copiar no es un n�mero entero v�lido');
   end
   
   if ceil(row) ~= row
       error('La fila de inicio para copiar no es un n�mero entero v�lido');
   end
   
%    siz = size(color);
%    
%    if length(siz) ~=2 || siz(1) ~= 1 || siz(2) ~=3
%        error('El color proporcionado no es un color RGB v�lido')
%    end
%    
%    if any(ceil(color) ~= color)  || any(color < 0 | color > 255)
%        error('El color proporcionado no es un color RGB v�lido');
%    end

   if isscalar(Rcolor) == 0 || isscalar(Gcolor) == 0 || isscalar(Bcolor) == 0
       error('Los valores RGB del color introducido deben de ser n�mero enteros')
   end
   
   if ceil(Rcolor) ~= Rcolor || Rcolor < 0 || Rcolor >255
       error('Rcolor no representa una tonalidad de rojo v�lida');
   end
   
   if ceil(Gcolor) ~= Gcolor || Gcolor < 0 || Gcolor >255
       error('Gcolor no representa una tonalidad de verde v�lida');
   end
   
   if ceil(Bcolor) ~= Bcolor || Bcolor < 0 || Bcolor >255
       error('Bcolor no representa una tonalidad de azul v�lida');
   end
   
   if any(sizF > sizB)
       error('El tama�o de la imagen Foreground debe de menor o igual a la imagen de Background');
   end
   
   if nargin > 8
       error('S�lo est� permitido expresar "eps" como par�metro extra');
   elseif nargin == 8
       eps = varargin{1};
   else
       eps = 0.12; % Valor por defecto
   end

   if isscalar(eps) == 0
       error('El par�metro extra "eps" debe ser un n�mero real');
   end
   
   if eps < 0 || eps > 1
       error('El par�metro extra "eps" debe ser un n�mero real entre 0 y 1');
   end
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Procedimiento                 %
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
   color = [Rcolor, Gcolor, Bcolor];
   
   % Copiar el color al objeto auxilizar y transformarlo a HSV
   rgb_color = zeros(1,1,3);
   rgb_color(1,1,:) = color;
   
   hsv_color = rgb2hsv(rgb_color);
   
   h = hsv_color(1,1,1); % Tomar Hue
   
   % Imagen auxiliar
   Iaux = zeros(sizB, 'uint8');
   Iaux(:,:,1) = color(1);
   Iaux(:,:,2) = color(2);
   Iaux(:,:,3) = color(3);
   
   %row_limit = (min(sizF(1),sizB(1)));
   %col_limit = (min(sizF(2),sizB(2)-col));
   
   Iaux(row:min(sizF(1)+row-1, sizB(1)), col:min(sizF(2)+col-1, sizB(2)), :) = F(1:min(sizF(1), sizB(1)-row+1), 1:min(sizF(2), sizB(2)-col+1), :);
   
   % Convertir la imagen a hsv
   F_hsv = rgb2hsv(Iaux);
   
   % Desplazar el matiz de la imagen hasta colocar el color a transparentar
   %  en el �ngulo 1
   aux = mod(F_hsv(:,:,1) + (0.5 - h), 1.0);
   aux = 1 - 2*abs(0.5-aux);
   
   % Generar la m�scara (todo valor igual a 1 se corresponde con el color
   % a eliminar)
   mask = uint8(aux < 1-eps);
   mask = repmat(mask,1,1,3);
   inv_mask = uint8(~mask);
   
   disp(size(mask));
   disp(size(F));
   % Superponer la imagen al fondo
   result = B.*inv_mask + Iaux.*mask; %B&(~mask) | F&mask
   
end