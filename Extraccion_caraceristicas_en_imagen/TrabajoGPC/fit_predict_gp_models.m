function prob_test = fit_predict_gp_models(pos_train_data, neg_train_data, pos_test_data, neg_test_data, pos_label, neg_label, mod_per_pos, initial_hyp, inf_method, mean_func, cov_func, lik_func, n_evals)
    
    % Agrupar los datos de test
    data_test = [pos_test_data; neg_test_data];
    labels_test = [repmat(1, size(pos_test_data,1), 1);
                    repmat(-1, size(neg_test_data,1), 1)];

    %   Normalizar datos
    data_test = norm_dataset(data_test);
                
                
    % Para almacenar las probabilidades devueltas por cada modelo
    probs = zeros(size(labels_test,1), mod_per_pos); % Resultados de los modelos
                
    % Construir varios modelos enfrentando las instancias de la clase
    % positiva con subconjuntos de la clase negativa
    part_size = floor(size(neg_train_data,1)/mod_per_pos); % Tamaño partición
    
    for i=1:part_size:mod_per_pos*part_size
        
        % Partir el conjunto de instancias negativas en "mod_per_pos"
        % conjuntos disjuntos, si la división no fuese exacta, las
        % instancias restantes se colocan en la última partición
        if i < (mod_per_pos-1)*part_size
            neg_data_sub = neg_train_data(i:i+part_size-1,:);
        else
            neg_data_sub = neg_train_data(i:end,:);
        end

        % Ajustar hiperparámetros
        data = [pos_train_data;neg_data_sub];
        labels = [repmat(pos_label, size(pos_train_data,1), 1);
                        repmat(neg_label, size(neg_data_sub,1), 1)];
        
        %   Normalizar dataset
        data = norm_dataset(data);
        
        hyp = minimize(initial_hyp, @gp, n_evals, inf_method, mean_func, cov_func, lik_func, data, labels); %hyp_models(i,:)
        
        % Ajustar y tomar la predicción de cada modelo
        [a, b, c, d, lp] = gp(hyp, inf_method, mean_func, cov_func, lik_func, data, labels, data_test, ones(size(data_test,1),1));
        probs(:,floor(i/part_size)+1) = exp(lp);

    end

    % Calcular probabilidad media de los modelos
    %disp(probs)
    prob_test = mean(probs, 2);
    
end