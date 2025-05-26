# 🎓 Student Performance Prediction Model

This project uses a **Multiple Linear Regression** model to predict student academic performance based on various behavioral and academic factors. The dataset includes a range of student habits, and the model aims to determine how these factors influence performance outcomes.

## 🧠 Overview

- 🗂️ Dataset: `enhanced_student_habits_performance_dataset.csv`
- 🧮 Model: Multiple Linear Regression
- 🎯 Goal: Predict student performance and evaluate feature influence
- 📊 Visualizations: Heatmap correlation, scatter plot of actual vs predicted values

## 📦 Features & Workflow

### ✅ Step-by-Step Breakdown

1. **Import Libraries**: `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`
2. **Load Dataset**: Reads student performance data into a DataFrame.
3. **Preprocess Data**:
   - Encode categorical variables using `LabelEncoder` and `OneHotEncoder`
   - Avoid the dummy variable trap by excluding one dummy column
4. **Split Dataset**:
   - 80% training, 20% testing (`train_test_split`)
5. **Train Model**: Fits a `LinearRegression` model on training data
6. **Make Predictions**: Predicts on test data
7. **Evaluate Model**:
   - Prints coefficients and intercept
   - Outputs R² score (`r2_score`)
   - Visualizes predictions vs. actual results

## 📈 Visualizations

- 🔥 **Correlation Heatmap** of numerical features
- 📉 **Scatter Plot** comparing actual vs predicted performance

## 🛠 Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Install required libraries with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# ml-student-performance
