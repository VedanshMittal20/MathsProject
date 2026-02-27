// Scilab script to train simple models on the exported numeric datasets.
// Requires files created by running: python src/export_for_scilab.py

funcprot(0); // allow re-defining helper funcs when re-running

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
core = csvRead(core_path, ",", ".", "double"); // read as numbers
core = core(2:$, :); // drop header row
y = core(:, 1);
X_raw = core(:, 2:$);
// standardize features to improve GD convergence
muX = mean(X_raw, "r");
sdX = stdev(X_raw, "r");
sdX(find(sdX==0)) = 1;
X_std = (X_raw - ones(size(X_raw,1),1)*muX) ./ (ones(size(X_raw,1),1)*sdX);
X = [ones(size(X_std, 1), 1) X_std]; // add intercept

[theta_logit, loss_logit] = logisticGD(X, y, 0.0005, 6000);
probs = sigmoid(X * theta_logit);
threshold = mean(y); // start from class balance
pred = probs >= threshold;
// if degenerate (all same), fallback to 0.5 threshold
if sum(pred)==0 | sum(pred)==length(pred) then
    threshold = 0.5;
    pred = probs >= threshold;
end
acc = sum(pred == y) / length(y);
disp("Logistic regression (GD) accuracy:");
disp(acc);
disp("Positive class rate (threshold used):");
disp(threshold);
disp("Mean(pred):");
disp(mean(double(pred)));

// Frequency modeling: simple log-linear regression on accidents
freq_path = "data/processed/frequency_dataset_numeric.csv";
freq = csvRead(freq_path, ",", ".", "double");
freq = freq(2:$, :); // drop header row
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

// ---------- Visualization section ----------
fig = 0;

// 1) Speed limit distribution
fig = fig + 1; scf(fig); clf(); histplot(20, core(:,2)); xtitle("Speed limit distribution","km/h","count");

// 2) Speed vs severity (boxplots)
sev = core(:,1); speed = core(:,2);
// boxplot is not always available by default; implement a minimal two-group box plot
function q = percentile(v, p)
    v = gsort(v, "g", "i");
    n = size(v, "*");
    m = size(p, "*");
    q = zeros(m,1);
    for i = 1:m
        k = (p(i)/100) * (n-1) + 1;
        k1 = floor(k); k2 = ceil(k);
        if k1 == k2 then
            q(i) = v(k1);
        else
            q(i) = v(k1) + (k - k1) * (v(k2) - v(k1));
        end
    end
endfunction
function simple_box(vals, xpos, width, col)
    q = percentile(vals, [25 50 75]);
    iqr = q(3) - q(1);
    low = max(min(vals), q(1) - 1.5*iqr);
    high = min(max(vals), q(3) + 1.5*iqr);
    xrects([xpos - width/2; q(1); width; iqr]);
    plot([xpos - width/2, xpos + width/2], [q(2) q(2)], col);
    plot([xpos xpos], [q(3) high], col);
    plot([xpos - width/4, xpos + width/4], [high high], col);
    plot([xpos xpos], [q(1) low], col);
    plot([xpos - width/4, xpos + width/4], [low low], col);
endfunction
fig = fig + 1; scf(fig); clf(); xgrid();
simple_box(speed(sev==0), 1, 0.5, "b");
simple_box(speed(sev==1), 2, 0.5, "r");
xtitle("Speed vs severity","class","km/h");
ax = gca();
ax.x_ticks = tlist(["ticks","locations","labels"], [1 2], ["Non-severe" "Severe"]);

// 3) Accidents by hour (bar histogram)
hours = core(:,7);
fig = fig + 1; scf(fig); clf(); histplot(0:23, hours); xtitle("Accidents by hour","hour","count");

// 4) Monthly accident trend (line plot)
months = freq(:,2); acc = freq(:,3);
month_avg = zeros(12,1);
for m = 1:12
    idx = find(months==m);
    if size(idx,"*") > 0 then
        month_avg(m) = mean(acc(idx));
    else
        month_avg(m) = 0;
    end
end
fig = fig + 1; scf(fig); clf();
plot(1:12, month_avg, "-o");
xtitle("Monthly accidents","month","count");
xgrid();

// 5) Severe rate by rain/night (2x2 heatmap)
rain = core(:,11); night = core(:,9);
Z = zeros(2,2);
for r=0:1
    for n=0:1
        idx = find(rain==r & night==n);
        if size(idx,"*") > 0 then
            Z(r+1,n+1) = mean(sev(idx));
        else
            Z(r+1,n+1) = %nan;
        end
    end
end
Z(find(isnan(Z))) = 0;
fig = fig + 1; scf(fig); clf();
Matplot(Z);
colorbar();
xtitle("Severe rate by rain/night","Rain (0/1)","Night (0/1)");
for i=1:2
    for j=1:2
        xstring(i-0.3, j, sprintf("%.2f", Z(j,i)));
    end
end
ax = gca();
ax.x_ticks = tlist(["ticks","locations","labels"], [1 2], ["0" "1"]);
ax.y_ticks = tlist(["ticks","locations","labels"], [1 2], ["0" "1"]);

// 6) Correlation matrix of core features
function R = corrmat(M)
    [rows, cols] = size(M);
    mu = mean(M, "r");
    sigma = stdev(M, "r");
    // avoid divide-by-zero
    sigma(find(sigma==0)) = %eps;
    Z = (M - ones(rows,1)*mu) ./ (ones(rows,1)*sigma);
    R = (Z' * Z) / (rows - 1);
endfunction
R = corrmat(core(:,2:$)); // correlation matrix
fig = fig + 1; scf(fig); clf();
Matplot(R);
colorbar();
xtitle("Correlation matrix","feature index","feature index");
// annotate a few correlations (diagonal omitted for clarity)
for i=1:12
    for j=1:12
        if i<>j then
            xstring(i-0.35, j, sprintf("%.2f", R(j,i)));
        end
    end
end

// 7) Vehicles vs accidents scatter (frequency dataset)
veh = freq(:,5);
fig = fig + 1; scf(fig); clf(); plot(veh, acc, "."); xtitle("Vehicles_{avg} vs accidents","vehicles_{avg}","accidents");

// 8) Logistic GD loss curve
fig = fig + 1; scf(fig); clf(); plot(loss_logit); xtitle("Logistic GD loss","epoch","cross-entropy");

// 9) ROC curve (simple implementation)
function [fpr,tpr,thr] = simpleROC(labels, scores)
    thr = gsort(scores, "g", "i");
    thr = unique([thr; 0]);
    m = size(thr, "*");
    P = sum(labels==1);
    N = sum(labels==0);
    fpr = zeros(m,1);
    tpr = zeros(m,1);
    for k = 1:m
        preds = scores >= thr(k);
        tp = sum(preds==1 & labels==1);
        fp = sum(preds==1 & labels==0);
        tpr(k) = tp / P;
        fpr(k) = fp / N;
    end
endfunction
[fpr,tpr,thr] = simpleROC(sev, probs);
// ensure curve starts at (0,0) and ends at (1,1)
fpr = [0; fpr; 1];
tpr = [0; tpr; 1];
fig = fig + 1; scf(fig); clf();
plot(fpr, tpr, "-b");
plot([0 1],[0 1], "r--");
xtitle("ROC curve","FPR","TPR");
xgrid();

// 10) Calibration plot (pred prob bins vs observed severe rate)
bins = 0:0.1:1; obs = zeros(1,10); mid = bins(1:$-1) + 0.05;
for k=1:10
    idx = find(probs>=bins(k) & probs<bins(k+1));
    if size(idx,"*") > 0 then
        obs(k) = mean(sev(idx));
    else
        obs(k) = %nan;
    end
end
fig = fig + 1; scf(fig); clf(); plot(mid, obs, "-o"); plot(mid, mid, "r--");
xtitle("Calibration","pred prob","observed severe rate");

// 11) Predicted vs actual counts (frequency model)
fig = fig + 1; scf(fig); clf(); plot(y_freq, pred_cnt, "."); xtitle("Predicted vs actual accidents","actual","predicted");

// 12) Residuals vs fitted (frequency model)
res = pred_cnt - y_freq;
fig = fig + 1; scf(fig); clf(); plot(pred_cnt, res, "."); xtitle("Residuals vs fitted","predicted","residual");

// 13) Confusion matrix for severity classes
pred_num = double(pred);
cm = [sum(pred_num==0 & sev==0), sum(pred_num==1 & sev==0);
      sum(pred_num==0 & sev==1), sum(pred_num==1 & sev==1)];
disp("Confusion matrix [TN FP; FN TP]:");
disp(cm);
disp("Class counts (y=0, y=1):");
disp([sum(sev==0), sum(sev==1)]);
clf();
// keep default colormap for portability
Matplot(cm);
colorbar();
xtitle("Confusion matrix","Pred 0/1","True 0/1");
// overlay counts
for i=1:2
    for j=1:2
        xstring(i-0.2, j, string(cm(j,i)));
    end
end
