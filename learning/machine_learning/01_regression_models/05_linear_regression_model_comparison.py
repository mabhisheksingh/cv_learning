import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

## load the diabetes regression dataset
## X contains input features and Y contains the target value we want to predict
X, Y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(Y.shape)

## split the dataset into training data and test data
## the model learns only from the training set
## the test set is kept separate so we can evaluate on unseen examples
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"X train shape {X_train.shape}")
print(f"X test shape {X_test.shape}")
print(f"Y train shape {Y_train.shape}")
print(f"Y test shape {Y_test.shape}")

## create a simple linear regression model
## this model tries to learn a straight-line relationship between inputs and target
model = LinearRegression()
model.fit(X_train, Y_train)

## coefficients are the learned weights for each feature
## intercept is the bias term added to the prediction
print(f"w = {model.coef_}")
print(f"b = {model.intercept_}")

## evaluate the simple linear regression model on both train and test data
## train score tells us how well the model fits seen data
## test score tells us how well the model generalizes to unseen data
train_pred = model.predict(X_train)
Y_pred = model.predict(X_test)
linear_train_r2 = r2_score(Y_train, train_pred)
linear_mse = mean_squared_error(Y_test, Y_pred)
linear_rmse = np.sqrt(linear_mse)
linear_mae = mean_absolute_error(Y_test, Y_pred)
linear_r2 = r2_score(Y_test, Y_pred)

## cross-validation gives a more reliable estimate than a single train/test split
## here sklearn splits the training set into 5 folds and trains 5 times
## each time it uses 4 folds for training and 1 fold for validation
linear_cv_scores = cross_val_score(LinearRegression(), X_train, Y_train, cv=5, scoring="r2")

print("\n--- Simple Linear Regression ---")
print(f"Train R² = {linear_train_r2:.4f}")
print(f"Test R² = {linear_r2:.4f}")
print(f"5-fold CV R² mean = {linear_cv_scores.mean():.4f}")
print(f"5-fold CV R² std = {linear_cv_scores.std():.4f}")
print(f"MSE = {linear_mse:.2f}")
print(f"RMSE = {linear_rmse:.2f}")
print(f"MAE = {linear_mae:.2f}")

## baseline comparison: predict the mean target value for every example
## a useful model should perform better than this naive baseline
mean_pred = np.full_like(Y_test, Y_train.mean())
baseline_mse = mean_squared_error(Y_test, mean_pred)
print(f"Baseline (mean) MSE = {baseline_mse:.2f}")
print(f"Model improvement over baseline = {(baseline_mse - linear_mse) / baseline_mse * 100:.1f}%")

## polynomial features add non-linear terms such as x^2 and x1*x2
## this lets a linear model learn curved relationships in a transformed feature space
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(f"\n--- Polynomial Features (degree=2) ---")
print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {X_train_poly.shape[1]}")

## fit linear regression on the expanded polynomial feature set
model_poly = LinearRegression()
model_poly.fit(X_train_poly, Y_train)

poly_train_pred = model_poly.predict(X_train_poly)
Y_pred_poly = model_poly.predict(X_test_poly)
poly_train_r2 = r2_score(Y_train, poly_train_pred)
poly_mse = mean_squared_error(Y_test, Y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_mae = mean_absolute_error(Y_test, Y_pred_poly)
poly_r2 = r2_score(Y_test, Y_pred_poly)
poly_cv_scores = cross_val_score(LinearRegression(), X_train_poly, Y_train, cv=5, scoring="r2")

print(f"Train R² = {poly_train_r2:.4f}")
print(f"Test R² = {poly_r2:.4f}")
print(f"5-fold CV R² mean = {poly_cv_scores.mean():.4f}")
print(f"5-fold CV R² std = {poly_cv_scores.std():.4f}")
print(f"MSE = {poly_mse:.2f}")
print(f"RMSE = {poly_rmse:.2f}")
print(f"MAE = {poly_mae:.2f}")
print(f"Improvement over simple linear = {poly_r2 - linear_r2:.4f}")

## scale polynomial features because regularized models like Ridge are sensitive to feature magnitude
## fit_transform learns scaling values from the training set and applies them
## transform reuses the same scaling values for the test set
poly_scaler = StandardScaler()
X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)
X_test_poly_scaled = poly_scaler.transform(X_test_poly)

## Ridge regression is linear regression with L2 regularization
## alpha controls the strength of the penalty on large coefficients
## RidgeCV automatically tests many alpha values and selects the best one by cross-validation
alpha_values = np.logspace(-3, 3, 13)
model_ridge = RidgeCV(alphas=alpha_values, cv=5)
model_ridge.fit(X_train_poly_scaled, Y_train)

ridge_train_pred = model_ridge.predict(X_train_poly_scaled)
Y_pred_ridge = model_ridge.predict(X_test_poly_scaled)
ridge_train_r2 = r2_score(Y_train, ridge_train_pred)
ridge_mse = mean_squared_error(Y_test, Y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae = mean_absolute_error(Y_test, Y_pred_ridge)
ridge_r2 = r2_score(Y_test, Y_pred_ridge)
ridge_cv_scores = cross_val_score(RidgeCV(alphas=alpha_values, cv=5), X_train_poly_scaled, Y_train, cv=5, scoring="r2")

print(f"\n--- Polynomial Features + Ridge ---")
print(f"Best alpha = {model_ridge.alpha_}")
print(f"Train R² = {ridge_train_r2:.4f}")
print(f"Test R² = {ridge_r2:.4f}")
print(f"5-fold CV R² mean = {ridge_cv_scores.mean():.4f}")
print(f"5-fold CV R² std = {ridge_cv_scores.std():.4f}")
print(f"MSE = {ridge_mse:.2f}")
print(f"RMSE = {ridge_rmse:.2f}")
print(f"MAE = {ridge_mae:.2f}")
print(f"Improvement over simple linear = {ridge_r2 - linear_r2:.4f}")

## collect metrics for comparison plots
model_names = ["Linear", "Poly", "Poly+Ridge"]
rmse_scores = [linear_rmse, poly_rmse, ridge_rmse]
mae_scores = [linear_mae, poly_mae, ridge_mae]
r2_scores = [linear_r2, poly_r2, ridge_r2]

## choose the best model based on the highest test R² score
best_index = int(np.argmax(r2_scores))
best_name = model_names[best_index]
best_predictions = [Y_pred, Y_pred_poly, Y_pred_ridge][best_index]

print(f"\nBest model based on R²: {best_name}")

## create plots to visually inspect prediction quality and compare metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

## plot actual target values against predicted target values
## points close to the red diagonal line indicate better predictions
axes[0].scatter(Y_test, best_predictions, color="royalblue", alpha=0.7)
axes[0].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red")
axes[0].set_title(f"Actual vs Predicted ({best_name})")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")

## residuals are prediction errors: actual - predicted
## a good model tends to have residuals scattered around zero without a strong pattern
residuals = Y_test - best_predictions
axes[1].scatter(best_predictions, residuals, color="darkorange", alpha=0.7)
axes[1].axhline(0, color="red")
axes[1].set_title(f"Residual Plot ({best_name})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residual")

## compare evaluation metrics across all models
x = np.arange(len(model_names))
width = 0.2
axes[2].bar(x - width, rmse_scores, width, label="RMSE")
axes[2].bar(x, mae_scores, width, label="MAE")
axes[2].bar(x + width, r2_scores, width, label="R²")
axes[2].set_xticks(x)
axes[2].set_xticklabels(model_names)
axes[2].set_title("Metric Comparison")
axes[2].legend()

plt.tight_layout()
plt.show()
