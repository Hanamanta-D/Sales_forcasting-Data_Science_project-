#!/usr/bin/env python
"""
Test script for Baseline Model Training
Runs only baseline model training and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Quick test of baseline model functions
def calculate_mape(y_true, y_pred):
    """Calculate MAPE"""
    valid_mask = y_true != 0
    if valid_mask.sum() == 0:
        return 0.0
    mape = np.mean(np.abs((y_true[valid_mask] - y_pred[valid_mask]) / y_true[valid_mask])) * 100
    return mape

# Create dummy data for testing
np.random.seed(42)
n_train = 1000
n_test = 250

X_train = np.random.randn(n_train, 5)
y_train = 5 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(n_train) * 10 + 100

X_test = np.random.randn(n_test, 5)
y_test = 5 * X_test[:, 0] + 3 * X_test[:, 1] + np.random.randn(n_test) * 10 + 100

print("\n" + "="*70)
print("BASELINE MODEL TRAINING TEST")
print("="*70)

# Step 1 & 2: Train Mean Baseline
print("\nStep 1 & 2: Training Mean Baseline Model...")
baseline_pred_value = y_train.mean()
predictions_mean = np.full_like(y_test, baseline_pred_value, dtype=float)
print(f"   Baseline prediction (constant): ${baseline_pred_value:.2f}")

# Step 3: Evaluate Mean Baseline
print("\nStep 3: Evaluating Mean Baseline Performance...")
rmse_mean = np.sqrt(mean_squared_error(y_test, predictions_mean))
mae_mean = mean_absolute_error(y_test, predictions_mean)
mape_mean = calculate_mape(y_test, predictions_mean)
r2_mean = r2_score(y_test, predictions_mean)

print(f"   RMSE: ${rmse_mean:.2f}")
print(f"   MAE: ${mae_mean:.2f}")
print(f"   MAPE: {mape_mean:.2f}%")
print(f"   R²: {r2_mean:.4f}")

# Train Linear Regression
print("\nTraining Linear Regression (Advanced Model)...")
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, predictions_lr))
mae_lr = mean_absolute_error(y_test, predictions_lr)
mape_lr = calculate_mape(y_test, predictions_lr)
r2_lr = r2_score(y_test, predictions_lr)

print(f"   RMSE: ${rmse_lr:.2f}")
print(f"   MAE: ${mae_lr:.2f}")
print(f"   MAPE: {mape_lr:.2f}%")
print(f"   R²: {r2_lr:.4f}")

# Step 4: Document comparison
print("\n" + "="*70)
print("STEP 4: BASELINE MODEL DOCUMENTATION")
print("="*70)

print("\n1. BASELINE MODEL SELECTED:")
print("   Model: Mean Sales (Average)")
print("   Why: Reference point for measuring advanced model improvements")

print("\n2. BASELINE MODEL PERFORMANCE METRICS:")
print(f"   • RMSE: ${rmse_mean:.2f}")
print(f"   • MAE: ${mae_mean:.2f}")
print(f"   • MAPE: {mape_mean:.2f}%")
print(f"   • R² Score: {r2_mean:.4f}")

print("\n3. BASELINE vs ADVANCED MODELS COMPARISON:")
print(f"   {'Model':<30} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
print("   " + "-"*65)
print(f"   {'Mean Sales Baseline':<30} ${rmse_mean:<11.2f} ${mae_mean:<11.2f} {r2_mean:<9.4f}")
print(f"   {'Linear Regression':<30} ${rmse_lr:<11.2f} ${mae_lr:<11.2f} {r2_lr:<9.4f}")

improvement_pct = ((rmse_mean - rmse_lr) / rmse_mean) * 100
print(f"\n   Improvement (RMSE): {improvement_pct:+.1f}%")

print("\n4. KEY INSIGHTS:")
print(f"   • Baseline RMSE: ${rmse_mean:.2f} (constant mean prediction)")
print(f"   • Advanced RMSE: ${rmse_lr:.2f} (Linear Regression)")
print(f"   • Performance Gain: {improvement_pct:.1f}% improvement")
print(f"   • Conclusion: Advanced model performs {abs(improvement_pct):.1f}% better")

print("\n5. BASELINE MODEL INTERPRETATION:")
print("   • Baseline predicts constant value (average of training sales)")
print("   • Assumption: Future sales = average of historical sales")
print("   • Limitations:")
print("     - Does not capture seasonal patterns")
print("     - Does not account for features")
print("     - Cannot adapt to trend changes")
print("     - Serves only as reference point")

print("\n" + "="*70)
print("Baseline Model Test Completed Successfully!")
print("="*70)
