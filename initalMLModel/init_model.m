clear
clc

traindata = readtable("train_data.csv"); % reads imported table from py

features = {'Close', 'Volume', 'High', 'Low', 'Open'}; 

x = traindata{:, features}; % input
y = traindata.Target; % output

% scaling
mu = mean(x);
sigma = std(x, 1);

xscaled = (x - mu) ./ sigma;

% tables for model
tbltrain = array2table(xscaled, 'variableNames', features);
tbltrain.Target = y;

% model
logregs = fitglm( tbltrain, ...
    'Target ~ Close + Volume + High + Low + Open', ...
    'Distribution', 'binomial');

% pulling coeffs for py
coeffs = logregs.Coefficients.Estimate;

b = coeffs(1); % intercepts
w = coeffs(2:end); % weights

% makes a table for csv file
paramnames = ["intercept"; string(features(:)); "mu_" + string(features(:)); "sigma_" + string(features(:))];
paramvalues = [b; w; mu(:); sigma(:)]; 

paramstable = table(paramnames, paramvalues, 'variableNames', {'parameter', 'value'});

writetable(paramstable, "matlab_logistic_params.csv");

%% testing and visualization

testData = readtable("test_data.csv");

Xtest = testData{:, features};
ytest = testData.Target;

% scale test data
Xtest_scaled = (Xtest - mu) ./ sigma;

% test data into table format
tblTest = array2table(Xtest_scaled, 'VariableNames', features);
tblTest.Target = ytest;

%pPredicted probability of class 1 = UP
probUp = predict(logregs, tblTest);

% convert probabilities to class predictions using 0.5 threshold
ypred = double(probUp >= 0.5);

% accuracy
accuracy = mean(ypred == ytest);
disp("Accuracy: " + string(accuracy))

% confusion matrix
figure
cm = confusionchart(ytest, ypred);
cm.Title = "Confusion Matrix: Logistic Regression";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";

% ROC curve and AUC
[Xroc, Yroc, thresholds, AUC] = perfcurve(ytest, probUp, 1);

figure
plot(Xroc, Yroc, 'LineWidth', 2)
hold on
plot([0 1], [0 1], '--', 'LineWidth', 1)
hold off

grid on
xlabel("False Positive Rate")
ylabel("True Positive Rate")
title("ROC Curve: Logistic Regression")
legend("ROC Curve, AUC = " + string(AUC), "Random Classifier", ...
    "Location", "southeast")