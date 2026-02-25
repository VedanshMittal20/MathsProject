// Scilab script to train simple models on the exported numeric datasets.
// Requires files created by running: python src/export_for_scilab.py

// Helper: sigmoid
function y = sigmoid(z)
    y = 1 ./(1 + exp(-z));
endfunction

// Logistic regression via gradient descent
function [theta, history] = logisticGD(X, y, alpha, epochs)
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    history = zeros(epochs, 1);
    for k = 1:epochs
        h = sigmoid(X * theta);
        grad = (1/m) * (X' * (h - y));
        theta = theta - alpha * grad;
        // cross-entropy loss
        history(k) = - (1/m) * (y' * log(h + %eps) + (1 - y)' * log(1 - h + %eps));
    end
endfunction

// Load compact numeric dataset (13 columns, first is severity_binary)
core_path = "data/processed/accidents_core_numeric.csv";
core = csvRead(core_path, ",", ".", 1); // skip header row
y = core(:, 1);
X = core(:, 2:$);
X = [ones(size(X, 1), 1) X]; // add intercept

[theta_logit, loss_logit] = logisticGD(X, y, 0.0005, 4000);
probs = sigmoid(X * theta_logit);
pred = probs >= 0.5;
acc = sum(pred == y) / length(y);
disp("Logistic regression (GD) accuracy:");
disp(acc);

// Frequency modeling: simple log-linear regression on accidents
freq_path = "data/processed/frequency_dataset_numeric.csv";
freq = csvRead(freq_path, ",", ".", 1);
// Columns: state_code, month_num, accidents, avg_speed_limit, vehicles_avg,
// casualties_avg, fatalities_avg, rainy_frac, night_frac, alcohol_frac
y_freq = freq(:, 3);
Xf = freq(:, 4:$);
Xf = [ones(size(Xf, 1), 1) Xf];
// model log(accidents + 1) to approximate Poisson
y_log = log(y_freq + 1);
beta = Xf \ y_log;
pred_log = Xf * beta;
pred_cnt = exp(pred_log) - 1;
// naive RMSE
rmse = sqrt(sum((pred_cnt - y_freq).^2) / length(y_freq));
disp("Log-linear freq RMSE:");
disp(rmse);

// Save coefficients for inspection
csvWrite(theta_logit', "outputs/model_metrics/scilab_theta_logit.csv");
csvWrite(beta', "outputs/model_metrics/scilab_beta_freq.csv");
