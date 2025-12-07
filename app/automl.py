import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def identify_issues(df):
    """Returns a summary of missing values and duplicates."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    duplicates = df.duplicated().sum()
    return {"missing_values": missing, "duplicates": int(duplicates)}

def auto_clean(df):
    """
    Intelligently cleans the dataset:
    - Drops duplicates
    - Fills numeric missing values with Median
    - Fills categorical missing values with Mode
    """
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Drop Duplicates
    df.drop_duplicates(inplace=True)
    
    # 2. Fill Missing Values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            
    log = f"Cleaned data. Dropped {initial_rows - len(df)} duplicates. Filled missing values."
    return df, log

def auto_encode(df):
    """
    Encodes categorical variables so they can be used in ML models.
    """
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            # Convert to string to handle mixed types safely
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
    return df, "Encoded categorical columns: " + ", ".join(encoders.keys())

def find_best_model(df, target_col, problem_type=None):
    """
    Runs a model tournament and PLOTS Feature Importance.
    """
    # 1. Setup X and y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Detect Problem Type
    if problem_type is None:
        if y.nunique() < 20 or y.dtype == 'object':
            problem_type = "classification"
        else:
            problem_type = "regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    best_model = None
    best_score = -999
    
    # 3. Define Models
    if problem_type == "regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
        }
        metric_name = "R2 Score"
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
        }
        metric_name = "Accuracy"

    # 4. Train and Evaluate
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
            else:
                score = accuracy_score(y_test, predictions)
                
            results.append({"Model": name, metric_name: round(score, 4)})
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
        except Exception as e:
            results.append({"Model": name, metric_name: "Failed"})

    # 5. Generate Feature Importance Plot (The "Explain" Part)
    plt.figure(figsize=(10, 6))
    
    importances = None
    if "Random Forest" in best_model_name:
        importances = best_model.feature_importances_
    elif "Linear" in best_model_name or "Logistic" in best_model_name:
        importances = np.abs(best_model.coef_)
        if importances.ndim > 1: # Handle multi-class logistic regression
             importances = np.mean(importances, axis=0)

    if importances is not None:
        indices = np.argsort(importances)[-10:] # Top 10 features
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title(f'Feature Importance ({best_model_name})')
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model", 
                 ha='center', va='center')

    results_df = pd.DataFrame(results).sort_values(by=metric_name, ascending=False)
    
    return results_df, f"Best model was {best_model_name} with {metric_name} = {best_score:.4f}. (See plot for details)"