clc; clear;

% Carga de datos
data = load('./Datos');

healthy_folds = data.Healthy_folds;
malign_folds = data.Malign_folds;

ell = 1.9; % Escala inicial del núcleo SE
sf = 1.0; % Varianza inicial del núcleo SE

thres = 0.5; % Umbral de probabilidad para clasificar

% Ajustar y evaluar rendimiento del modelo sobre cada fold
mean_accuracy = 0.0;

% Datos de la curva ROC y curva Precission-Recall
%ROC_curve = zeros(5, 4);
%PR_curve = zeros(5, 4);

for i=1:5
    
    fprintf('Fold nº %d\n', i);
    
    tic
    % Tomar los conjuntos de entrenamiento y test en este fold
    train_healthy = healthy_folds(setdiff(1:5,i));
    train_healthy = [train_healthy(1).histogram;
                    train_healthy(2).histogram;
                    train_healthy(3).histogram;
                    train_healthy(4).histogram];
    test_healthy = healthy_folds(i).histogram;
    
    train_malign = malign_folds(setdiff(1:5,i));
    train_malign = [train_malign(1).histogram;
                    train_malign(2).histogram;
                    train_malign(3).histogram;
                    train_malign(4).histogram];
    test_malign = malign_folds(i).histogram;
    
    % Designar valores iniciales para los hiperparámetros
    initial_hyp.cov = log([ell, sf]);
    
    % Ajustar los hiperparámetros, ajustar el modelo y predecir el test
    prob = fit_predict_gp_models(train_malign, train_healthy, test_malign, test_healthy, 1, -1, 4, initial_hyp, @infVB, @meanZero, @covSEiso, @likLogistic, -40);
    
    % Ajustar modelos de clasificación basados en GP y predecir test
    %y_train = [repmat(1, size(train_healthy,1), 1); repmat(-1, size(train_malign,1), 1)];
    y_test = [repmat(1, size(test_malign,1), 1);
                repmat(-1, size(test_healthy,1), 1)];
    
    %prob = fit_predict_gp_models(hyp, @infVB, @meanZero, @covSEiso, @likLogistic, [train_healthy;train_malign], y_train, [test_healthy;test_malign], y_test);

    % Considerando un umbral de probabilidad de "thres"
    y_pred = zeros(size(prob));
    y_pred(prob >= thres) = 1;
    y_pred(prob < thres) = -1;
    
    % Calcular matriz de confusión y otras metricas de rendimiento
    score(i) = meassure_accuracy(y_test, y_pred);
    mean_accuracy = mean_accuracy + score(i).accuracy;
    
    % Calcular la curva Precission-Recall y el área bajo la curva
    [X1, Y1, T, AUC] = perfcurve(y_test, prob, 1);
    ROC_curve(i) = struct('X', X1, 'Y', Y1, 'T', T, 'AUC', AUC);
    
    % Calcular la curva ROC y el área bajo la curva
    [X2, Y2, T2pr, AUC2] = perfcurve(y_test, prob, 1, 'xCrit', 'sens', 'yCrit', 'prec');
    PR_curve(i) = struct('X', X2, 'Y', Y2, 'T', T2pr, 'AUC', AUC2);
    toc
    
end

disp('Métricas por fold');
disp(score);

mean_accuracy = mean_accuracy/5;
fprintf('Accuracy medio del modelo sobre los folds: %f\n', mean_accuracy);

i_fold = 1;

hold on
area(ROC_curve(i_fold).X, ROC_curve(i_fold).Y, 'FaceColor', [0.3010, 0.7450, 0.9330]);
plot(ROC_curve(i_fold).X, ROC_curve(i_fold).Y, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 3);
text(0.7,0.1, sprintf('AUC=%f', ROC_curve(i_fold).AUC), 'Color', [0.3922, 0.3922, 0.3922], 'Fontsize', 12)
line('Color','red', 'Color',[0.8500, 0.3250, 0.0980], 'LineWidth', 3);
%plot(T(1), T(2), 'r*');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Curva ROC - Fold %d', i_fold));
hold off

hold on
area(PR_curve(i_fold).X, PR_curve(i_fold).Y, 'FaceColor', [0.3010, 0.7450, 0.9330]);
text(0.1,0.1, sprintf('AUC=%f', PR_curve(i_fold).AUC), 'Color', [0.3922, 0.3922, 0.3922], 'Fontsize', 12)
plot(PR_curve(i_fold).X, PR_curve(i_fold).Y, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 3);
line([0,1],[1,0],'Color',[0.8500, 0.3250, 0.0980], 'LineWidth', 3);
%plot(T(1), T(2), 'r*');
xlabel('Recall');
ylabel('Precision');
title(sprintf('Curva Precision-Recall - Fold %d', i_fold));
hold off