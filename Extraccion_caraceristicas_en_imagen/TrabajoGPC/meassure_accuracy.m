function score = meassure_accuracy(y_test, y_pred)

    % Calcular la matriz de confusión de la matriz
    conf_matrix = confusionmat(y_test, y_pred, 'Order', [max(y_test), min(y_test)]);
    disp(conf_matrix);
    % Calcular el resto de métricas
    accuracy = (conf_matrix(1,1) + conf_matrix(2,2))/sum(sum(conf_matrix));
    specificity = conf_matrix(2,2)/(conf_matrix(2,2)+conf_matrix(2,1));
    sensitivity = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(1,2));
    precision = conf_matrix(1,1)/(conf_matrix(1,1)+conf_matrix(2,1));
    
    f_score = 2*precision*sensitivity/(precision+sensitivity);
    
    % Devolver las métricas conjuntamente
    score = struct('confussion_matrix', conf_matrix, 'accuracy', accuracy, 'specificity', specificity, 'sensitivity', sensitivity, 'precision', precision, 'f_score', f_score);
    
end