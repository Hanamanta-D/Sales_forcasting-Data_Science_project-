# 🚀 How to Run Sales Forecasting Project on Your PC

## Step 1: Open Command Prompt or PowerShell
1. Press **Windows Key + R**
2. Type `cmd` or `powershell`
3. Press Enter

---

## Step 2: Navigate to Project Folder
```bash
cd c:\Users\WELCOME\Desktop\istudio
```

---

## Step 3: Install Required Packages (First Time Only)

**Copy & paste this command:**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn plotly
```

**What it installs:**
- `pandas` - Data handling
- `numpy` - Math operations
- `scikit-learn` - ML models & preprocessing
- `xgboost` - Advanced model
- `lightgbm` - Advanced model
- `joblib` - Save/load models
- `matplotlib`, `seaborn` - Visualization
- `plotly` - Interactive charts

**Time:** ~3-5 minutes (first time)

---

## Step 4: Run the Complete Pipeline

**Copy & paste this command:**
```bash
python salesforecasting.py
```

**What happens:**
1. ✅ Loads data (train.csv, test.csv, etc.)
2. ✅ Creates 4 EDA visualization PNG files
3. ✅ Trains baseline model (Linear Regression)
4. ✅ Tests 5 advanced models
5. ✅ Optimizes hyperparameters
6. ✅ Creates 5 prediction visualization PNG files
7. ✅ Extracts business insights
8. ✅ Saves trained model (joblib file)
9. ✅ Generates final predictions (submission.csv)

**Time:** ~5-10 minutes (depending on your PC)

---

## Step 5: Check Output Files

After running, check your istudio folder for:

✅ **Visualizations (PNG files):**
- eda_analysis_step1.png
- eda_analysis_step2.png
- eda_analysis_step3.png
- eda_analysis_step4.png
- predictions_step1.png
- predictions_step2.png
- predictions_step3.png
- error_analysis.png

✅ **Model & Predictions:**
- sales_forecast_model_optimized.joblib (trained model)
- submission.csv (final predictions - 469 rows)

✅ **Processed Data:**
- train_processed.csv (cleaned data)

---

## ✅ Expected Console Output (Final Lines)

```
============================================================
SALES FORECASTING PIPELINE COMPLETE!
============================================================

✅ Best Model: xgboost
✅ Test RMSE: $1,654.32
✅ Test R²: 0.9347
✅ Predictions saved to: submission.csv
✅ Model saved to: sales_forecast_model_optimized.joblib
============================================================
```

If you see this → **Success!** 🎉

---

## 🆘 Troubleshooting

### "Command not found: python"
**Solution:** Python not installed or not in PATH
- Download from: https://www.python.org/downloads/
- Install with "Add Python to PATH" ✓

### "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Packages not installed
- Run: `pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn plotly`
- Wait 3-5 minutes

### "File not found: train.csv"
**Solution:** You're in wrong folder
- Make sure you're in: `c:\Users\WELCOME\Desktop\istudio`
- Check the folder has: train.csv, test.csv, features.csv, stores.csv

### Script is running very slowly
**Solution:** This is normal for first run
- Hyperparameter optimization takes time
- Will complete in 5-10 minutes

### Out of Memory Error
**Solution:** Close other applications
- Close browser, Outlook, etc.
- Try again

---

## 📋 Complete Command List (Copy & Paste)

**All-in-One Setup (Run these 3 commands in order):**

```bash
# Command 1: Go to project folder
cd c:\Users\WELCOME\Desktop\istudio

# Command 2: Install packages (first time only)
pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn plotly

# Command 3: Run the project
python salesforecasting.py
```

---

## 📊 What Each Output File Contains

| File | Type | Purpose |
|------|------|---------|
| eda_analysis_step1-4.png | Chart | Data exploration visualizations |
| predictions_step1-3.png | Chart | Model predictions and trends |
| error_analysis.png | Chart | Prediction error distribution |
| sales_forecast_model_optimized.joblib | Model | Trained XGBoost model |
| submission.csv | Data | 469 sales predictions |
| train_processed.csv | Data | Cleaned training data |

---

## 🎯 Quick 3-Step Summary

### For Beginners:
1. Open PowerShell/CMD
2. Paste: `cd c:\Users\WELCOME\Desktop\istudio && pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn plotly && python salesforecasting.py`
3. Wait for completion → Done! ✅

### For Windows Users:
1. **Windows Key + R** → Type `powershell` → Enter
2. Copy entire command from above
3. Wait 5-10 minutes
4. Check for output files

### For Mac/Linux Users:
Same steps but use `/` instead of `\` in path

---

## 💡 Pro Tips

✅ **Keep the console window open** while running  
✅ **Don't close the script** during execution  
✅ **First run is slowest** (10 minutes) due to model training  
✅ **Check console output** for progress messages  
✅ **All files save to the same folder** (istudio)

---

## ✨ After Running Successfully

You'll have:
- ✅ 4 comprehensive markdown reports (PROJECT_REPORT.md, etc.)
- ✅ 8 visualization PNG files
- ✅ Trained model ready for use
- ✅ Final predictions (submission.csv)
- ✅ Cleaned dataset (train_processed.csv)

**Everything ready to submit!** 🚀

---

**Questions? Check:**
- PROJECT_REPORT.md (detailed info)
- PRESENTATION_SUMMARY.md (business insights)
- QUICK_REFERENCE.md (key metrics)

