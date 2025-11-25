# 📊 AI Data Intelligence Dashboard

### *Power BI–Style Analytics • Streamlit • Machine Learning • Data Chatbot*

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)

![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)

![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)

![License](https://img.shields.io/badge/License-MIT-green)

![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📌 Overview

This project is a **Power BI–style Data Analytics Dashboard** built with  **Streamlit** , offering a complete end-to-end data workflow:

✔ Upload

✔ Clean

✔ Visualize

✔ Train ML models

✔ Predict

✔ Chat with your dataset

All in one interactive web application — no coding required for the user.

---

## ✨ Features

### 📂 **1. Upload & Explore Data**

* Supports CSV, Excel, and JSON
* Auto-detection of datatypes
* Null analysis, unique counts, memory usage
* Clean UI and detailed info tables

---

### 🧹 **2. Data Cleaning Module**

* Remove duplicates
* Drop or fill missing values
* Fill using mean/median/mode
* Intelligent data quality report
* Cleaned dataframe stored separately

---

### 📈 **3. Visualizations (Fully Customizable)**

* Line Chart
* Bar Chart
* Scatter Plot
* Histogram
* Box Plot
* Pie Chart
* Multiple themes & interactive elements
* Save charts as PNG

---

### 🤖 **4. Machine Learning**

#### Regression:

* Linear Regression
* Random Forest Regressor
* Metrics: MSE, RMSE, R²

#### Classification:

* Logistic Regression
* Random Forest Classifier
* Metrics: Accuracy, Classification Report

Features:

* Automatic preprocessing
* One-hot encoding
* Predict on new inputs
* Smart evaluation metrics

---

### 💬 **5. AI Chat with Your Data**

Ask natural language questions like:

* “Give me insights about this data”
* “Show correlation matrix”
* “Any outliers in my dataset?”
* “Analyze column age”
* “What cleaning steps should I take?”

The assistant intelligently analyzes:

* Numeric & categorical columns
* Distribution
* Outliers
* Correlations
* Cleaning recommendations
* ML feature quality

---

### 📥 **6. Export Cleaned Dataset**

Download in:

* CSV
* Excel
* JSON

---

### ℹ️ **7. About Page**

Includes:

* Project overview
* Tech stack
* Best practices
* ML model explanations
* User tips

---

## 🛠️ Tech Stack

| Technology             | Usage                |
| ---------------------- | -------------------- |
| **Streamlit**    | UI Framework         |
| **Pandas**       | Data handling        |
| **NumPy**        | Numerical operations |
| **Matplotlib**   | Visualization        |
| **Scikit-Learn** | ML models            |
| **OpenPyXL**     | Excel export         |

---

## 📁 Project Structure

```
📦 AI Data Dashboard
 ┗━━ main.py        # Complete dashboard application
```

---

# ▶️ Installation Guide

## **1️⃣ Install dependencies (requirements.txt)**

If you have a `requirements.txt`, install everything with:

```bash
pip install -r requirements.txt
```

### 🔍 If pip shows permission errors:

```bash
pip install --user -r requirements.txt
```

### 🔍 If Streamlit or sklearn fails to install:

```bash
pip install --upgrade pip setuptools wheel
```

---

## **2️⃣ Run the Streamlit app**

```bash
streamlit run main.py
```

### Open the app in your browser:

```
http://localhost:8501
```

---

# ⚡ Windows Setup Script (.bat File)

Below is the improved version of your `.bat` script with clean formatting.
Save it as: **setup_and_run.bat**

```bat
@echo off
REM ============================================================
REM  Make Any Windows System Compatible for Streamlit Project
REM  Includes: Virtual Environment, Dependencies, Run App
REM  Author: M. Shah Nawaz
REM ============================================================

echo.
echo ===========================================
echo   Setting up Environment For AI Dashboard
echo ===========================================
echo.

REM 1) Create virtual environment
python -m venv .venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python is installed and added to PATH.
    pause
    exit /b
)

REM 2) Activate virtual environment
call .venv\Scripts\activate

echo.
echo Virtual environment activated.
echo.

REM 3) Upgrade pip
python -m pip install --upgrade pip

REM 4) Install required dependencies
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: One or more dependencies failed to install.
    pause
    exit /b
)

echo.
echo All dependencies installed successfully!
echo.

REM 5) Run Streamlit app
echo Starting Streamlit Application...
streamlit run "%~dp0main.py"

echo.
echo Application stopped.
pause









