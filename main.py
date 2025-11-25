import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
import io
warnings.filterwarnings('ignore')

# --------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------
#============================================================================================================
#----------- preprocessing data ----------------------------
def preprocess_data(df):
    df = df.copy()
    
    # Convert date columns
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
            df.drop(columns=[col], inplace=True)

    # Boolean → int
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Handle categorical → OUTSIDE the loop
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Handle missing numeric
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df

#----------------------- Overview of Data ------------------------------------
def show_overview_cards(df):
    """Display KPI cards with dataset metrics"""
    try:
        total_records = len(df)
        total_features = len(df.columns)
        missing_values = df.isnull().sum().sum()
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{total_records:,}")
        with c2:
            st.metric("Features", f"{total_features}")
        with c3:
            st.metric("Missing Values", f"{missing_values}")
        with c4:
            st.metric("Numeric Columns", f"{numeric_cols}")
    except Exception as e:
        st.error(f"Error displaying overview: {e}")
#---------------------------- Categorical data to Numeric --------------------------
def safe_convert_numeric(df, columns):
    """Safely convert columns to numeric"""
    for col in columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            st.warning(f"Could not convert {col} to numeric: {e}")
    return df
# ------------------- Intelligent query analyzer that understands context and provides detailed answers -----------------
def analyze_query_and_respond(df, query):
    query = query.lower().strip()
    
    try:
        # Advanced pattern matching for intelligent responses
        
        # Dataset overview and insights
        if any(word in query for word in ["insight", "tell me about", "overview", "analyze", "summary of data", "what can you tell"]):
            numeric_df = df.select_dtypes(include=[np.number])
            categorical_df = df.select_dtypes(include=['object'])
            
            result = "## 📊 Comprehensive Dataset Analysis\n\n"
            result += f"**Structure:** {len(df):,} rows × {len(df.columns)} columns\n\n"
            
            # Data completeness
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            result += f"**Data Quality:** {completeness:.1f}% complete ({missing_cells:,} missing values)\n\n"
            
            # Numeric insights
            if not numeric_df.empty:
                result += f"### 📈 Numeric Analysis ({len(numeric_df.columns)} columns)\n\n"
                for col in numeric_df.columns[:5]:
                    mean_val = numeric_df[col].mean()
                    std_val = numeric_df[col].std()
                    cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                    result += f"**{col}:**\n"
                    result += f"- Range: [{numeric_df[col].min():.2f}, {numeric_df[col].max():.2f}]\n"
                    result += f"- Mean ± Std: {mean_val:.2f} ± {std_val:.2f}\n"
                    result += f"- Coefficient of Variation: {cv:.1f}%\n\n"
            
            # Categorical insights
            if not categorical_df.empty:
                result += f"### 🏷️ Categorical Analysis ({len(categorical_df.columns)} columns)\n\n"
                for col in categorical_df.columns[:3]:
                    nunique = categorical_df[col].nunique()
                    mode_val = categorical_df[col].mode()[0] if len(categorical_df[col].mode()) > 0 else "N/A"
                    mode_count = (categorical_df[col] == mode_val).sum()
                    result += f"**{col}:** {nunique} categories, most frequent: '{mode_val}' ({mode_count} times)\n"
            
            return result
        
        # Specific column analysis
        for col in df.columns:
            if col.lower() in query and any(word in query for word in ["about", "analyze", "tell", "show", "describe"]):
                result = f"## 📋 Detailed Analysis: '{col}'\n\n"
                result += f"**Data Type:** {df[col].dtype}\n"
                result += f"**Completeness:** {df[col].count()}/{len(df)} values ({df[col].count()/len(df)*100:.1f}%)\n"
                result += f"**Unique Values:** {df[col].nunique()} distinct\n"
                result += f"**Missing:** {df[col].isnull().sum()} values\n\n"
                
                if df[col].dtype in ['int64', 'float64']:
                    result += "### 📊 Statistical Summary\n\n"
                    result += f"| Metric | Value |\n|--------|-------|\n"
                    result += f"| Mean | {df[col].mean():.4f} |\n"
                    result += f"| Median | {df[col].median():.4f} |\n"
                    result += f"| Std Dev | {df[col].std():.4f} |\n"
                    result += f"| Min | {df[col].min():.4f} |\n"
                    result += f"| 25% | {df[col].quantile(0.25):.4f} |\n"
                    result += f"| 75% | {df[col].quantile(0.75):.4f} |\n"
                    result += f"| Max | {df[col].max():.4f} |\n"
                    result += f"| Range | {df[col].max() - df[col].min():.4f} |\n"
                    result += f"| Variance | {df[col].var():.4f} |\n\n"
                    
                    # Outlier detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
                    outliers = len(df[outlier_mask])
                    
                    if outliers > 0:
                        result += f"### ⚠️ Outlier Detection\n\n"
                        result += f"Found **{outliers}** potential outliers ({outliers/len(df)*100:.1f}% of data)\n"
                        result += f"- Lower bound: {Q1 - 1.5 * IQR:.2f}\n"
                        result += f"- Upper bound: {Q3 + 1.5 * IQR:.2f}\n\n"
                    
                    # Distribution
                    skew = df[col].skew()
                    result += f"### 📉 Distribution\n\n"
                    result += f"**Skewness:** {skew:.3f} "
                    if abs(skew) < 0.5:
                        result += "(approximately symmetric)\n"
                    elif skew > 0:
                        result += "(right-skewed, tail extends right)\n"
                    else:
                        result += "(left-skewed, tail extends left)\n"
                else:
                    result += "### 🏷️ Category Distribution\n\n"
                    top_values = df[col].value_counts().head(10)
                    result += "| Value | Count | Percentage |\n|-------|-------|------------|\n"
                    for val, count in top_values.items():
                        pct = (count / len(df)) * 100
                        result += f"| {val} | {count} | {pct:.1f}% |\n"
                
                return result
        
        # Correlation analysis
        if "correlat" in query or "relationship" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                result = "## 🔗 Correlation Analysis\n\n"
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        corr_val = corr.iloc[i, j]
                        if abs(corr_val) > 0.3:  # Lowered threshold
                            strong_corr.append((corr.columns[i], corr.columns[j], corr_val))
                
                if strong_corr:
                    result += "### 📊 Significant Correlations\n\n"
                    result += "| Feature 1 | Feature 2 | Correlation | Strength |\n"
                    result += "|-----------|-----------|-------------|----------|\n"
                    for col1, col2, val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                        strength = "Very Strong" if abs(val) > 0.8 else "Strong" if abs(val) > 0.6 else "Moderate"
                        result += f"| {col1} | {col2} | {val:.3f} | {strength} |\n"
                else:
                    result += "No significant correlations (>0.3) found between numeric columns.\n"
                
                result += "\n**Interpretation:**\n"
                result += "- Values close to 1: Strong positive relationship\n"
                result += "- Values close to -1: Strong negative relationship\n"
                result += "- Values close to 0: Little to no linear relationship\n"
                
                return result
            return "❌ Need at least 2 numeric columns for correlation analysis."
        
        # Comparison queries
        if "compare" in query or "difference between" in query or "versus" in query or "vs" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                result = "## ⚖️ Numeric Columns Comparison\n\n"
                result += "| Column | Mean | Median | Std Dev | Min | Max | Range |\n"
                result += "|--------|------|--------|---------|-----|-----|-------|\n"
                for col in numeric_df.columns:
                    result += f"| {col} | {numeric_df[col].mean():.2f} | {numeric_df[col].median():.2f} | "
                    result += f"{numeric_df[col].std():.2f} | {numeric_df[col].min():.2f} | "
                    result += f"{numeric_df[col].max():.2f} | {numeric_df[col].max() - numeric_df[col].min():.2f} |\n"
                return result
            return "❌ No numeric columns available for comparison."
        
        # Outlier detection
        if "outlier" in query or "anomal" in query or "extreme" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                result = "## 🎯 Outlier Detection Analysis\n\n"
                result += "Using IQR method (1.5 × IQR beyond quartiles)\n\n"
                result += "| Column | Outliers | Percentage | Lower Bound | Upper Bound |\n"
                result += "|--------|----------|------------|-------------|-------------|\n"
                
                has_outliers = False
                for col in numeric_df.columns:
                    Q1 = numeric_df[col].quantile(0.25)
                    Q3 = numeric_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = len(numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)])
                    
                    if outliers > 0:
                        has_outliers = True
                        pct = outliers/len(df)*100
                        result += f"| {col} | {outliers} | {pct:.1f}% | {lower:.2f} | {upper:.2f} |\n"
                
                if not has_outliers:
                    result += "| - | No outliers detected | - | - | - |\n"
                
                return result
            return "❌ No numeric columns to check for outliers."
        
        # Distribution queries
        if "distribut" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                result = "## 📊 Distribution Analysis\n\n"
                result += "| Column | Skewness | Interpretation | Kurtosis |\n"
                result += "|--------|----------|----------------|----------|\n"
                
                for col in numeric_df.columns:
                    skew = numeric_df[col].skew()
                    kurt = numeric_df[col].kurtosis()
                    
                    if abs(skew) < 0.5:
                        interp = "Symmetric"
                    elif skew > 0:
                        interp = "Right-skewed"
                    else:
                        interp = "Left-skewed"
                    
                    result += f"| {col} | {skew:.3f} | {interp} | {kurt:.3f} |\n"
                
                result += "\n**Notes:**\n"
                result += "- Skewness: measures asymmetry (0 = symmetric)\n"
                result += "- Kurtosis: measures tail heaviness (3 = normal)\n"
                
                return result
            return "❌ No numeric columns for distribution analysis."
        
        # Cleaning recommendations
        if "clean" in query or "should i" in query or "recommend" in query or "improve" in query:
            issues = []
            recommendations = []
            
            # Check duplicates
            dup = df.duplicated().sum()
            if dup > 0:
                issues.append(f"✗ **{dup}** duplicate rows found ({dup/len(df)*100:.1f}%)")
                recommendations.append("→ Remove duplicates using the Clean page")
            
            # Check missing values
            missing = df.isnull().sum()
            high_missing = missing[missing > len(df) * 0.5]
            if len(high_missing) > 0:
                issues.append(f"✗ **{len(high_missing)}** columns with >50% missing data")
                recommendations.append(f"→ Consider dropping: {', '.join(high_missing.index.tolist())}")
            
            moderate_missing = missing[(missing > 0) & (missing <= len(df) * 0.5)]
            if len(moderate_missing) > 0:
                issues.append(f"✗ **{len(moderate_missing)}** columns with missing values")
                recommendations.append(f"→ Fill missing in: {', '.join(moderate_missing.index.tolist())}")
            
            # Check data types
            object_cols = df.select_dtypes(include=['object']).columns
            convertible = []
            for col in object_cols:
                try:
                    pd.to_numeric(df[col].dropna(), errors='raise')
                    convertible.append(col)
                except:
                    pass
            
            if convertible:
                issues.append(f"✗ **{len(convertible)}** columns could be numeric")
                recommendations.append(f"→ Convert to numeric: {', '.join(convertible)}")
            
            # Build result
            result = "## 🧹 Data Quality Assessment\n\n"
            
            if issues:
                result += "### ⚠️ Issues Found:\n\n"
                for issue in issues:
                    result += f"{issue}\n"
                
                result += "\n### 💡 Recommendations:\n\n"
                for rec in recommendations:
                    result += f"{rec}\n"
            else:
                result += "### ✅ Data Quality: Excellent!\n\n"
                result += "No major issues detected. Your data is ready for analysis!"
            
            return result
        
        # Feature selection for ML
        if "feature" in query or "machine learning" in query or "ml model" in query or "prediction" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                result = "## 🤖 Machine Learning Feature Analysis\n\n"
                
                # Check variance
                variances = numeric_df.var().sort_values(ascending=False)
                result += "### 📊 Feature Variance (Higher is Better)\n\n"
                result += "| Feature | Variance | Recommendation |\n"
                result += "|---------|----------|----------------|\n"
                for col, var in variances.items():
                    rec = "Good" if var > variances.median() else "Consider removing"
                    result += f"| {col} | {var:.4f} | {rec} |\n"
                
                # Check correlations
                corr = numeric_df.corr()
                result += "\n### 🔗 Multicollinearity Check\n\n"
                high_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.9:
                            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
                
                if high_corr:
                    result += "⚠️ **Highly correlated features detected (consider removing one):**\n\n"
                    for col1, col2, val in high_corr:
                        result += f"- {col1} ↔ {col2}: {val:.3f}\n"
                else:
                    result += "✅ No problematic multicollinearity detected!\n"
                
                return result
            return "❌ No numeric columns available for feature analysis."
        
        # Count/size queries
        if any(word in query for word in ["how many", "count", "number of", "total"]):
            if "row" in query:
                return f"📊 The dataset contains **{len(df):,}** rows."
            elif "column" in query or "feature" in query:
                return f"📊 The dataset has **{len(df.columns)}** columns: {', '.join(df.columns.tolist())}"
            elif "missing" in query or "null" in query:
                total_missing = df.isnull().sum().sum()
                return f"📊 There are **{total_missing:,}** missing values in the dataset ({total_missing/(len(df)*len(df.columns))*100:.2f}% of all cells)"
            else:
                return f"📊 **Dataset Size:**\n- Rows: {len(df):,}\n- Columns: {len(df.columns)}\n- Total cells: {len(df) * len(df.columns):,}"
        
        # Fallback to basic handler
        return fallback_query_handler(df, query)
        
    except Exception as e:
        return f"❌ Error analyzing query: {str(e)}\n\n💡 Try asking a simpler question or check your data format."
#-------------------------------------Fallback handler for basic queries when AI is unavailable----------------------------------------------------------------------------------------------
def fallback_query_handler(df, query):
    """"""
    query = query.lower().strip()
    
    try:
        # Basic statistics queries
        if any(word in query for word in ["how many", "count", "total rows", "number of rows"]):
            return f"📊 The dataset contains **{len(df):,}** rows and **{len(df.columns)}** columns."
        
        elif "columns" in query or "features" in query or "what columns" in query:
            cols = ", ".join(df.columns.tolist())
            return f"📋 **Columns in dataset:** {cols}"
        
        elif "missing" in query or "null" in query:
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if len(missing_cols) > 0:
                result = "⚠️ **Missing values:**\n\n"
                for col, count in missing_cols.items():
                    result += f"- {col}: {count} ({count/len(df)*100:.1f}%)\n"
                return result
            else:
                return "✅ No missing values found in the dataset!"
        
        elif "summary" in query or "describe" in query or "statistics" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats = numeric_df.describe().round(2)
                result = "📈 **Summary Statistics:**\n\n"
                result += stats.to_string()
                return result
            else:
                return "No numeric columns to summarize."
        
        elif "mean" in query or "average" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                means = numeric_df.mean()
                result = "📊 **Mean values:**\n\n"
                for col, val in means.items():
                    result += f"- {col}: {val:.2f}\n"
                return result
            else:
                return "No numeric columns to calculate mean."
        
        elif "max" in query or "maximum" in query or "highest" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                maxs = numeric_df.max()
                result = "📈 **Maximum values:**\n\n"
                for col, val in maxs.items():
                    result += f"- {col}: {val:.2f}\n"
                return result
            else:
                return "No numeric columns to find maximum."
        
        elif "min" in query or "minimum" in query or "lowest" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                mins = numeric_df.min()
                result = "📉 **Minimum values:**\n\n"
                for col, val in mins.items():
                    result += f"- {col}: {val:.2f}\n"
                return result
            else:
                return "No numeric columns to find minimum."
        
        elif "unique" in query or "distinct" in query:
            result = "🔢 **Unique values per column:**\n\n"
            for col in df.columns:
                result += f"- {col}: {df[col].nunique()} unique values\n"
            return result
        
        elif "show" in query and "data" in query:
            return f"📄 **First few rows:**\n\n{df.head().to_string()}"
        
        elif "correlation" in query or "corr" in query:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr().round(2)
                return f"🔗 **Correlation Matrix:**\n\n{corr.to_string()}"
            else:
                return "Need at least 2 numeric columns for correlation analysis."
        
        # Column-specific queries
        for col in df.columns:
            if col.lower() in query:
                result = f"📊 **Information about '{col}':**\n\n"
                result += f"- Data type: {df[col].dtype}\n"
                result += f"- Non-null count: {df[col].count()}\n"
                result += f"- Missing values: {df[col].isnull().sum()}\n"
                result += f"- Unique values: {df[col].nunique()}\n"
                
                if df[col].dtype in ['int64', 'float64']:
                    result += f"- Mean: {df[col].mean():.2f}\n"
                    result += f"- Min: {df[col].min():.2f}\n"
                    result += f"- Max: {df[col].max():.2f}\n"
                    result += f"- Median: {df[col].median():.2f}\n"
                else:
                    top_values = df[col].value_counts().head(5)
                    result += f"\n**Top 5 values:**\n"
                    for val, count in top_values.items():
                        result += f"- {val}: {count} occurrences\n"
                
                return result
        
        # Default response
        return """💡 I can help you explore your data! Try asking:

**General Questions:**
- "How many rows are there?"
- "What columns do you have?"
- "Show me the data"
- "Give me a summary"

**Statistics:**
- "What's the average/mean?"
- "Show maximum/minimum values"
- "What are the missing values?"
- "Show unique values"
- "Calculate correlation"

**Column-Specific:**
- "Tell me about [column name]"
- "What's the average [column name]?"

Ask me anything about your dataset!"""
    
    except Exception as e:
        return f"❌ Error processing query: {e}"

# --------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------
st.set_page_config(page_title="Power BI Dashboard", layout="wide", page_icon="📊")

# Enhanced Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.4);
        transform: translateX(5px);
    }
    
    /* Active page button */
    [data-testid="stSidebar"] .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: none;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Headers */
    h1 {
        color: #1e3c72;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2, h3 {
        color: #2a5298;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
    }
    
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 15px;
        border: 2px dashed #667eea;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-assistant {
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar logo area */
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #1e3c72;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #1e3c72;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------------
# Sidebar header with logo
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; background: rgba(255,255,255,0.1); border-radius: 15px; margin-bottom: 1rem;">
        <h1 style="color: white; font-size: 2rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📊</h1>
        <h2 style="color: white; font-size: 1.3rem; margin: 0.5rem 0 0 0; font-weight: 700;">Power BI</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.3rem 0 0 0;">AI Data Intelligence</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🧭 Navigation")

# Initialize session state for current page
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# Navigation menu with buttons
menu = [
    ("🏠 Home", "home"),
    ("📂 Upload", "upload"),
    ("📊 Summary", "summary"),
    ("🧹 Clean", "clean"),
    ("Download Updated","download"),
    ("📈 Visualize", "visualize"),
    ("🤖 Predict", "predict"),
    ("💬 Chat", "chat"),
    ("ℹ️ About", "about")
]

for label, key in menu:
    # Check if this is the current page
    is_current = st.session_state.page == label
    
    if st.sidebar.button(
        label,
        key=f"nav_{key}",
        width='stretch',
        type="primary" if is_current else "secondary"
    ):
        st.session_state.page = label
        st.rerun()

choice = st.session_state.page

# Sidebar info section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Status")

if "df" in st.session_state:
    st.sidebar.markdown("""
        <div style="background: rgba(76, 175, 80, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #4CAF50;">
            <p style="margin: 0; color: white;"><strong>✅ Data Loaded</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.metric("📝 Rows", f"{len(st.session_state.df):,}")
    st.sidebar.metric("📋 Columns", f"{len(st.session_state.df.columns)}")
    
    if "cleaned_df" in st.session_state:
        st.sidebar.markdown("""
            <div style="background: rgba(33, 150, 243, 0.2); padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">
                <p style="margin: 0; color: white; font-size: 0.9rem;"><strong>🧹 Cleaned</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    if "model" in st.session_state:
        st.sidebar.markdown("""
            <div style="background: rgba(156, 39, 176, 0.2); padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;">
                <p style="margin: 0; color: white; font-size: 0.9rem;"><strong>🤖 Model Ready</strong></p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
        <div style="background: rgba(255, 152, 0, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #FF9800;">
            <p style="margin: 0; color: white;"><strong>⚠️ No Data</strong></p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: rgba(255,255,255,0.8);">Upload a file to start</p>
        </div>
    """, unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px;">
        <p style="margin: 0; font-size: 0.9rem; color: rgba(255,255,255,0.7);">Built with ❤️</p>
        <p style="margin: 0.3rem 0 0 0; font-weight: 600; color: white;">Streamlit + ML</p>
        <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: rgba(255,255,255,0.6);">Version 2.0</p>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# HOME PAGE
# --------------------------------------------------------------
if choice == "🏠 Home":
    st.title("🌟 BitBros Data Destroyer (but it actually cleans)")
    # col1, col2, col3 = st.columns([1, 2, 1])
    # with col2:
    #     st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    #     st.image("bitbros.png", width=250)
    #     st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Welcome to AI-Powered Data Intelligence Platform")
    col1, col2, col3 = st.columns([1, 2, 1])
   


    
    col1, col2 ,col3= st.columns([1, 2,3])
    
    with col1:
        st.markdown("""
        ### 🚀 Key Features:
        
        **📂 Data Management**
        - Upload CSV, Excel, or JSON files
        - Automatic data type detection
        - Preview and explore your data
        
        **🧹 Data Cleaning**
        - Remove duplicate records
        - Handle missing values
        - Data type conversion
        
        **📈 Visualization**
        - Multiple chart types
        - Interactive plots
        - Custom styling
        
        **🤖 Machine Learning**
        - Regression models (predict continuous values)
        - Classification models (predict categories)
        - Model performance metrics
        
        **💬 Chat with Data**
        - Natural language queries
        - Instant insights
        - Statistical summaries
        """)
    # with col2:
    #     st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    #     st.image("bitbros.png", width=250)
    #     st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        ### 📌 Quick Start
        
        1. **Upload** your data file
        2. **Clean** your data
        3. **Visualize** insights
        4. **Predict** outcomes
        5. **Chat** with your data
        
        ### 🛠️ Tech Stack
        - Streamlit
        - Pandas
        - NumPy
        - Matplotlib
        - Scikit-learn
        """)

# --------------------------------------------------------------
# UPLOAD PAGE
# --------------------------------------------------------------







elif choice == "📂 Upload":
    st.header("📂 Upload Your Dataset")
    
    file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "json"],
        help="Upload CSV, Excel, or JSON files"
    )

    if file:
        try:
            # Read file based on extension
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                st.success(f"✅ Successfully loaded CSV file: {file.name}")

            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
                st.success(f"✅ Successfully loaded Excel file: {file.name}")

            elif file.name.endswith(".json"):
                df = pd.read_json(file)
                st.success(f"✅ Successfully loaded JSON file: {file.name}")

            else:
                st.error("❌ Unsupported file format")
                st.stop()

            # Store in session state
            st.session_state.df = df
            st.session_state.original_df = df.copy()

            # Data Overview
            st.subheader("📋 Data Overview")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{df.shape[1]}")
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

            # Preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), width="stretch")

            # Column Information Table
            st.subheader("🔍 Column Information")
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null": df.count(),
                "Null Count": df.isnull().sum(),
                "Null %": (df.isnull().sum() / len(df) * 100).round(2),
                "Unique": df.nunique()
            })

            st.dataframe(info_df, width="stretch")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Please ensure your file is properly formatted.")

    else:
        st.info("👆 Upload a file to get started")




# --------------------------------------------------------------
# SUMMARY PAGE
# --------------------------------------------------------------
elif choice == "📊 Summary":
    st.header("📊 Data Summary & Statistics")
    
    # Get the appropriate dataframe
    if "cleaned_df" in st.session_state:
        df = st.session_state.cleaned_df
        st.info("📌 Showing summary of cleaned data")
    elif "df" in st.session_state:
        df = st.session_state.df
    else:
        st.warning("⚠️ Please upload a dataset first")
        st.stop()

    try:
        # Overview Cards
        show_overview_cards(df)
        
        st.markdown("---")
        
        # Numeric Statistics
        st.subheader("📈 Numerical Statistics")
        numeric_df = df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            desc = numeric_df.describe()
            st.dataframe(desc.style.format("{:.2f}"), width='stretch')
            
            # Additional statistics
            st.subheader("📊 Additional Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Variance**")
                st.dataframe(numeric_df.var().to_frame(name="Variance").style.format("{:.2f}"))
            
            with col2:
                st.markdown("**Skewness**")
                st.dataframe(numeric_df.skew().to_frame(name="Skewness").style.format("{:.2f}"))
        else:
            st.info("No numeric columns found in the dataset")
        
        # Categorical Statistics
        st.subheader("🏷️ Categorical Columns")
        categorical_df = df.select_dtypes(include=['object'])
        
        if not categorical_df.empty:
            cat_stats = pd.DataFrame({
                "Column": categorical_df.columns,
                "Unique Values": [categorical_df[col].nunique() for col in categorical_df.columns],
                "Most Frequent": [categorical_df[col].mode()[0] if len(categorical_df[col].mode()) > 0 else "N/A" for col in categorical_df.columns],
                "Frequency": [categorical_df[col].value_counts().iloc[0] if len(categorical_df[col]) > 0 else 0 for col in categorical_df.columns]
            })
            st.dataframe(cat_stats, width='stretch')
        else:
            st.info("No categorical columns found in the dataset")
            
    except Exception as e:
        st.error(f"❌ Error generating summary: {str(e)}")

# --------------------------------------------------------------
# CLEAN DATA PAGE
# --------------------------------------------------------------

elif choice == "🧹 Clean":
    st.header("🧹 Data Cleaning Tools")
    
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    try:
        df = st.session_state.df.copy()
        
        st.subheader("📋 Original Data Preview")
        st.dataframe(df.head(), width='stretch')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Cleaning Options")
            remove_dup = st.checkbox("Remove Duplicate Rows")
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["None", "Drop rows with any missing", "Drop rows with all missing", 
                 "Fill with mean (numeric only)", "Fill with median (numeric only)", "Fill with mode (all types)"]
            )
            
        with col2:
            st.subheader("📊 Data Quality")
            duplicates = df.duplicated().sum()
            missing = df.isnull().sum().sum()
            
            st.metric("Duplicate Rows", duplicates)
            st.metric("Total Missing Values", missing)
        
        if st.button("🚀 Apply Cleaning", type="primary"):
            cleaned_df = df.copy()
            changes = []
            
            # Remove duplicates
            if remove_dup and duplicates > 0:
                cleaned_df = cleaned_df.drop_duplicates()
                changes.append(f"✅ Removed {duplicates} duplicate rows")
            
            # Handle missing values
            if handle_missing != "None":
                if handle_missing == "Drop rows with any missing":
                    before = len(cleaned_df)
                    cleaned_df = cleaned_df.dropna()
                    changes.append(f"✅ Dropped {before - len(cleaned_df)} rows with missing values")
                
                elif handle_missing == "Drop rows with all missing":
                    before = len(cleaned_df)
                    cleaned_df = cleaned_df.dropna(how='all')
                    changes.append(f"✅ Dropped {before - len(cleaned_df)} rows with all missing values")
                
                elif handle_missing in ["Fill with mean (numeric only)", "Fill with median (numeric only)", "Fill with mode (all types)"]:
                    for col in cleaned_df.columns:
                        if cleaned_df[col].isnull().any():
                            if handle_missing == "Fill with mean (numeric only)":
                                if np.issubdtype(cleaned_df[col].dtype, np.number):
                                    cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
                            elif handle_missing == "Fill with median (numeric only)":
                                if np.issubdtype(cleaned_df[col].dtype, np.number):
                                    cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
                            elif handle_missing == "Fill with mode (all types)":
                                mode_val = cleaned_df[col].mode()
                                if len(mode_val) > 0:
                                    cleaned_df[col].fillna(mode_val[0], inplace=True)
                    changes.append(f"✅ Filled missing values using strategy: {handle_missing}")
            
            # Store cleaned data
            st.session_state.cleaned_df = cleaned_df
            
            # Show changes
            st.success("Data cleaning completed!")
            for change in changes:
                st.write(change)
            
            # Show cleaned preview
            st.subheader("✨ Cleaned Data Preview")
            st.dataframe(cleaned_df.head(), width='stretch')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Rows", len(df))
            with col2:
                st.metric("Cleaned Rows", len(cleaned_df))
                
    except Exception as e:
        st.error(f"❌ Error during cleaning: {str(e)}")

#------------------------------------------------------------------
# Download updated file
#----------------------------------------------------------------

elif choice == "Download Updated":
    st.subheader("✨ Cleaned Data Preview")

    # Check if cleaned data exists
    if "cleaned_df" in st.session_state:
        cleaned_df = st.session_state.cleaned_df
        st.dataframe(cleaned_df.head(), width='stretch')
    else:
        st.warning("⚠️ No cleaned data available. Please clean your dataset first.")
        st.stop()

    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original Rows", len(st.session_state.df))
    with col2:
        st.metric("Cleaned Rows", len(cleaned_df))

    # ---------------- DOWNLOAD OPTIONS ----------------
    st.subheader("📥 Download Cleaned Data")
    download_format = st.selectbox(
        "Choose download format",
        ["CSV", "Excel", "JSON"]
    )

    if st.button("Download"):
        if download_format == "CSV":
            csv = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name='cleaned_data.csv',
                mime='text/csv'
            )
        elif download_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                cleaned_df.to_excel(writer, index=False, sheet_name='CleanedData')

            output.seek(0)  # IMPORTANT

            st.download_button(
            label="📥 Download as Excel",
            data=output,
            file_name='cleaned_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        elif download_format == "JSON":
            json_str = cleaned_df.to_json(orient='records').encode('utf-8')
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name='cleaned_data.json',
                mime='application/json'
            )

# --------------------------------------------------------------
# VISUALIZE PAGE
# --------------------------------------------------------------
elif choice == "📈 Visualize":
    st.header("📈 Data Visualization")

    # Get DataFrame
    if "cleaned_df" in st.session_state:
        data = st.session_state.cleaned_df
    elif "df" in st.session_state:
        data = st.session_state.df
    else:
        st.warning("⚠️ Please upload data first")
        st.stop()

    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        if not numeric_cols:
            st.warning("⚠️ No numeric columns available")
            st.stop()

        # -------------------------------
        # 🎨 Color Themes
        # -------------------------------
        theme_colors = {
            "Blue":    {"color": "#1f77b4", "edge": "#0d3c61"},
            "Green":   {"color": "#2ca02c", "edge": "#145214"},
            "Purple":  {"color": "#9467bd", "edge": "#4b2a6e"},
            "Orange":  {"color": "#ff7f0e", "edge": "#b04e00"},
            "Dark":    {"color": "#444444", "edge": "#222222"},
        }

        col_theme, col_chart = st.columns(2)
        with col_theme:
            theme = st.selectbox("🎨 Chart Theme", list(theme_colors.keys()), index=0)
        colors = theme_colors[theme]

        with col_chart:
            chart_type = st.selectbox(
                "📊 Chart Type",
                ["Line", "Bar", "Scatter", "Histogram", "Box Plot", "Pie Chart"],
            )

        # Input selectors
        if chart_type in ["Line", "Bar", "Scatter"]:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols)
        elif chart_type in ["Histogram", "Box Plot"]:
            selected_col = st.selectbox("Select Column", numeric_cols)
        elif chart_type == "Pie Chart":
            if not categorical_cols:
                st.warning("⚠️ No categorical columns for pie chart")
                st.stop()
            selected_cat = st.selectbox("Select Category", categorical_cols)

        # Create chart
        if st.button("🎨 Generate Visualization", type="primary"):
            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "Bar":
                x_data = data[x_col].head(20)
                y_data = data[y_col].head(20)
                ax.bar(
                    range(len(x_data)),
                    y_data,
                    color=colors["color"],
                    edgecolor=colors["edge"],
                )
                ax.set_xticks(range(len(x_data)))
                ax.set_xticklabels([f"{v:.1f}" for v in x_data], rotation=45)
                ax.set_title(f"Bar Chart: {y_col} vs {x_col}")

            elif chart_type == "Line":
                ax.plot(
                    data[x_col],
                    data[y_col],
                    color=colors["color"],
                    linewidth=2,
                    marker="o",
                )
                ax.set_title(f"Line Chart: {y_col} vs {x_col}")

            elif chart_type == "Scatter":
                ax.scatter(
                    data[x_col],
                    data[y_col],
                    color=colors["color"],
                    alpha=0.7,
                    s=60,
                    edgecolor=colors["edge"],
                )
                ax.set_title(f"Scatter: {y_col} vs {x_col}")

            elif chart_type == "Histogram":
                ax.hist(
                    data[selected_col].dropna(),
                    bins=30,
                    color=colors["color"],
                    edgecolor=colors["edge"],
                    alpha=0.8,
                )
                ax.set_title(f"Histogram: {selected_col}")

            elif chart_type == "Box Plot":
                ax.boxplot(
                    data[selected_col].dropna(),
                    patch_artist=True,
                    boxprops=dict(facecolor=colors["color"], edgecolor=colors["edge"]),
                )
                ax.set_title(f"Box Plot: {selected_col}")

            elif chart_type == "Pie Chart":
                pie_data = data[selected_cat].value_counts()
                ax.pie(
                    pie_data,
                    labels=pie_data.index,
                    autopct="%1.1f%%",
                    colors=plt.cm.Set3.colors,
                    startangle=90,
                )
                ax.set_title(f"Pie Chart: {selected_cat}")

            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Show chart
            st.pyplot(fig)

            # -------------------------------
            # 📥 Save chart as PNG
            # -------------------------------
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                "📥 Download PNG",
                data=buf,
                file_name=f"{chart_type.replace(' ', '_')}.png",
                mime="image/png",
            )

            plt.close()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# --------------------------------------------------------------
# PREDICT PAGE (ML)
# --------------------------------------------------------------

elif choice == "🤖 Predict":
    st.header("🤖 Machine Learning Predictions")

    # Load dataset
    if "cleaned_df" in st.session_state:
        df = st.session_state.cleaned_df.copy()
    elif "df" in st.session_state:
        df = st.session_state.df.copy()
    else:
        st.warning("⚠️ Please upload data first")
        st.stop()

    st.subheader("🎯 Model Configuration")

   
    
    # ==========================================================
    # 2. SELECT TARGET
    # ==========================================================
    target_col = st.selectbox("Select Target Column (what to predict)", df.columns)

    # Auto-detect problem type
    if df[target_col].dtype == object:
        problem_type = "Classification"
    else:
        problem_type = st.radio("Problem Type", ["Regression", "Classification"])

    # Separate X and y
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Encode target if classification
    target_encoder = None
    if problem_type == "Classification" and y_raw.dtype == object:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)
    else:
        y = y_raw.copy()

    # Preprocess features
    X = preprocess_data(X_raw)

    # ==========================================================
    # 3. MODEL SELECTION
    # ==========================================================
    if problem_type == "Regression":
        model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest"])
    else:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])

    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
    random_state = st.number_input("Random Seed", value=42)

    # ==========================================================
    # 4. TRAIN MODEL
    # ==========================================================
    if st.button("🚀 Train Model", type="primary"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

            # Choose model
            if problem_type == "Regression":
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=150, random_state=random_state)
            else:
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=2000)
                else:
                    model = RandomForestClassifier(n_estimators=150, random_state=random_state)

            # Train
            model.fit(X_train, y_train)

            st.success("✅ Model trained successfully!")

            # Save session state
            st.session_state.model = model
            st.session_state.target_encoder = target_encoder
            st.session_state.X_columns = X.columns.tolist()
            st.session_state.preprocess_func = preprocess_data
            st.session_state.target_col = target_col

            # ==================================================
            # METRICS
            # ==================================================
            st.subheader("📊 Model Performance")

            if problem_type == "Regression":
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.metric("R² Score", f"{r2:.4f}")
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("MSE", f"{mse:.4f}")

            else:  # Classification
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                st.metric("Accuracy", f"{accuracy:.4f}")
                st.write(pd.DataFrame(
                    classification_report(y_test, y_pred, output_dict=True)
                ).transpose())

        except Exception as e:
            st.error(f"❌ Training Error: {str(e)}")

    # ==========================================================
    # 5. PREDICT NEW VALUES
    # ==========================================================
    if "model" in st.session_state:
        st.markdown("---")
        st.subheader("🔮 Predict on New Input")

        input_values = {}
        cols = st.columns(2)

        for i, col in enumerate(X_raw.columns):
            with cols[i % 2]:
                dtype = X_raw[col].dtype

                if dtype == object:
                    options = X_raw[col].dropna().unique().tolist()
                    input_values[col] = st.selectbox(col, options)

                elif dtype == bool:
                    input_values[col] = st.selectbox(col, [True, False])

                elif np.issubdtype(dtype, np.datetime64):
                    input_values[col] = st.date_input(col)

                elif np.issubdtype(dtype, np.number):
                    input_values[col] = st.number_input(col, value=float(X_raw[col].mean()))

                else:
                    input_values[col] = st.text_input(col)

        if st.button("🎯 Predict"):
            try:
                # Convert to DF
                input_df = pd.DataFrame([input_values])

                # Preprocess input
                preprocess = st.session_state.preprocess_func
                input_processed = preprocess(input_df)

                # Align columns
                for col in st.session_state.X_columns:
                    if col not in input_processed:
                        input_processed[col] = 0

                input_processed = input_processed[st.session_state.X_columns]

                # Predict
                prediction = st.session_state.model.predict(input_processed)[0]

                # Decode classification
                if st.session_state.target_encoder:
                    prediction = st.session_state.target_encoder.inverse_transform([int(prediction)])[0]

                st.success(f"🎉 Prediction: **{prediction}**")

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")








# --------------------------------------------------------------
# CHAT PAGE
# --------------------------------------------------------------
elif choice == "💬 Chat":
    st.header("💬 Chat with Your Data")
    st.markdown("### 🤖 Intelligent Data Assistant")
    
    if "cleaned_df" in st.session_state:
        data = st.session_state.cleaned_df
    elif "df" in st.session_state:
        data = st.session_state.df
    else:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    try:
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            # Add welcome message
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"""👋 Hello! I'm your intelligent data assistant. I've analyzed your dataset:

**📊 Quick Facts:**
- **{len(data):,}** rows × **{len(data.columns)}** columns
- **{len(data.select_dtypes(include=[np.number]).columns)}** numeric columns
- **{data.isnull().sum().sum()}** missing values total

**💡 I can help you with:**
- Deep insights and analysis
- Column-specific information  
- Correlation and relationship analysis
- Outlier detection
- Data cleaning recommendations
- Feature selection for ML
- Statistical summaries

**Ask me anything!** I understand natural language queries."""
            })
        
        # Display example queries in an expandable section
        with st.expander("💡 Example Questions"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 Analysis Questions:**
                - "Give me insights about this data"
                - "What can you tell me about [column]?"
                - "Are there any outliers?"
                - "Analyze the distribution"
                
                **🔍 Investigation:**
                - "What's the correlation between columns?"
                - "Compare all numeric columns"
                - "What's the relationship between X and Y?"
                """)
            
            with col2:
                st.markdown("""
                **🧹 Recommendations:**
                - "What cleaning steps should I take?"
                - "Which features are good for prediction?"
                - "Should I remove any columns?"
                
                **📈 Statistics:**
                - "Show me summary statistics"
                - "What are the missing values?"
                - "Calculate correlation matrix"
                """)
        
        # Chat display area
        chat_container = st.container()
        
        with chat_container:
            for idx, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196F3;">
                        <strong>🧑 You:</strong><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4CAF50;">
                        <strong>🤖 Assistant:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(message['content'])
        
        st.markdown("---")
        
        # Input area
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your question here:",
                placeholder="e.g., What are the key insights from this data?",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 📤", type="primary", width='stretch')
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Overview", width='stretch'):
                user_input = "Give me insights about this data"
                send_button = True
        
        with col2:
            if st.button("🔍 Outliers", width='stretch'):
                user_input = "Are there any outliers?"
                send_button = True
        
        with col3:
            if st.button("🧹 Clean Tips", width='stretch'):
                user_input = "What cleaning steps should I take?"
                send_button = True
        
        with col4:
            if st.button("🔗 Correlations", width='stretch'):
                user_input = "Show correlation analysis"
                send_button = True
        
        # Process input
        if send_button and user_input and user_input.strip():
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Show processing indicator
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    # Get intelligent response
                    response = analyze_query_and_respond(data, user_input)
                except Exception as e:
                    response = f"❌ Error processing your question: {str(e)}\n\nPlease try rephrasing your question."
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Rerun to display new messages
            st.rerun()
        
        # Bottom controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🗑️ Clear Chat", width='stretch'):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("📋 Data Summary", width='stretch'):
                summary = f"""## 📊 Dataset Summary

**Structure:**
- Rows: {len(data):,}
- Columns: {len(data.columns)}
- Memory: {data.memory_usage(deep=True).sum() / 1024:.2f} KB

**Columns:** {', '.join(data.columns.tolist())}

**Missing Values:** {data.isnull().sum().sum()} total

**Numeric Columns:** {', '.join(data.select_dtypes(include=[np.number]).columns.tolist())}

**Categorical Columns:** {', '.join(data.select_dtypes(include=['object']).columns.tolist())}
"""
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": summary
                })
                st.rerun()
        
        with col3:
            st.info("💡 Tip: Ask specific questions for detailed analysis!")
            
    except Exception as e:
        st.error(f"❌ Error in chat module: {str(e)}")
        st.info("💡 Try reloading the page or uploading your data again.")


# --------------------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------------------
elif choice == "ℹ️ About":
    st.header("ℹ️ About This Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This **Power BI-style Analytics Dashboard** is a comprehensive data analysis platform 
        built with modern Python libraries. It provides end-to-end data analytics capabilities 
        from data ingestion to machine learning predictions.
        
        ### 🛠️ Technology Stack
        
        - **Streamlit**: Web application framework
        - **Pandas**: Data manipulation and analysis
        - **NumPy**: Numerical computations
        - **Matplotlib**: Data visualization
        - **Scikit-learn**: Machine learning models
        
        ### ✨ Key Features
        
        **1. Data Management**
        - Support for CSV, Excel, and JSON formats
        - Automatic data type detection
        - Comprehensive data preview and exploration
        
        **2. Data Cleaning**
        - Duplicate removal
        - Multiple missing value handling strategies
        - Data quality metrics
        
        **3. Visualization**
        - Line, Bar, Scatter, Histogram, and Box plots
        - Customizable chart options
        - Professional styling
        
        **4. Machine Learning**
        - **Regression Models**: Linear Regression, Random Forest Regressor
        - **Classification Models**: Logistic Regression, Random Forest Classifier
        - Feature importance analysis
        - Model performance metrics
        
        **5. Interactive Chat**
        - Natural language queries
        - Instant data insights
        - Statistical summaries on demand
        
        ### 🎓 Use Cases
        
        - **Business Analytics**: Sales forecasting, customer segmentation
        - **Data Science**: Exploratory data analysis, model prototyping
        - **Education**: Learning data science concepts
        - **Research**: Quick data analysis and visualization
        
        ### 📊 Machine Learning Models Explained
        
        **Regression (Predicting Numbers)**
        - Use when predicting continuous values (prices, temperatures, scores)
        - Linear Regression: Simple, interpretable, best for linear relationships
        - Random Forest: Handles complex patterns, more accurate for non-linear data
        
        **Classification (Predicting Categories)**
        - Use when predicting categories (yes/no, types, classes)
        - Logistic Regression: Simple, fast, works well for binary classification
        - Random Forest: More powerful, handles multiple classes well
        
        ### 🚀 Getting Started
        
        1. Upload your dataset (CSV, Excel, or JSON)
        2. Explore and clean your data
        3. Visualize patterns and trends
        4. Train ML models for predictions
        5. Chat with your data for quick insights
        """)
    
    with col2:
        st.markdown("""
        ### 📌 Quick Tips
        
        **Data Upload**
        - Ensure proper formatting
        - Check for encoding issues
        - Remove special characters
        
        **Data Cleaning**
        - Always backup original data
        - Check missing value patterns
        - Remove duplicates first
        
        **Visualization**
        - Choose appropriate chart types
        - Limit data points for clarity
        - Use colors meaningfully
        
        **ML Predictions**
        - Clean data first
        - Select relevant features
        - Use 20-30% test data
        - Check model metrics
        
        **Chat Feature**
        - Ask specific questions
        - Use clear language
        - Explore different queries
        
        ### 📞 Support
        
        For issues or suggestions:
        - Check data format
        - Ensure sufficient data
        - Review error messages
        
        ### 🔄 Version
        
        **v2.0**
        - Enhanced error handling
        - ML prediction module
        - Chat with data feature
        - Improved UI/UX
        """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Start with the Upload page and explore your data step by step!")