function prob_test = fit_predict_gp_models(hyp, inf_method, mean_func, cov_func, lik_func, X_train, y_train, X_test, y_test)

    n_models = length(hyp);      % Número de modelos
    probs = zeros(size(y_test,1), n_models); % Resultados de los modelos
    
    % Ajustar y tomar la predicción de cada modelo
    for i=1:n_models
        result = gp(hyp(i), inf_method, mean_func, cov_func, lik_func, X_train, y_train, X_test, y_test);
        probs(:,i) = exp(result(end));
    end

    % Calcular probabilidad media de los modelos
    disp(probs)
    prob_test = mean(probs, 2);

end