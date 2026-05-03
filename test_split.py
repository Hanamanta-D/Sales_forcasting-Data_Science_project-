#!/usr/bin/env python
"""Test script for chronological data splitting."""

from salesforecasting import (load_data, clean_data, merge_datasets, 
                             engineer_features, select_features,
                             split_train_test_chronological, 
                             verify_split_integrity, 
                             document_split_strategy)

# Load and prepare data
train, test, stores, features = load_data('.')
train, test, stores, features = clean_data(train, test, stores, features)
train, test = merge_datasets(train, test, stores, features)

print("="*70)
print("TEST: CHRONOLOGICAL DATA SPLITTING")
print("="*70)

# Feature engineering
X, y, feature_columns = engineer_features(train)
selected_features = select_features(X, y, feature_columns, method='correlation')
X = X[selected_features]

print(f"\nData prepared:")
print(f"  Total records: {len(X)}")
print(f"  Features: {len(selected_features)}")

# Perform chronological split
print(f"\nPerforming chronological train-test split (80-20)...")
X_train, X_test, y_train, y_test, df_train, df_test = split_train_test_chronological(
    train, X, y, train_ratio=0.8
)

print(f"\nSplit results:")
print(f"  Train set: {X_train.shape}")
print(f"  Test set:  {X_test.shape}")

# Verify integrity
verify_split_integrity(df_train, df_test, X_train, X_test, y_train, y_test)

# Document strategy
document_split_strategy(train_ratio=0.8, df_train=df_train, df_test=df_test)

print("\n[OK] All tests completed successfully!")
