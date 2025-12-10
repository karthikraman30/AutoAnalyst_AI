import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

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
    Runs a model tournament (Linear vs RF vs GradientBoosting) 
    and PLOTS Feature Importance.
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
    best_model_name = ""
    
    # 3. Define Models (Now with Gradient Boosting & Scaling Pipelines)
    # Note: Tree models (RF, GB) don't strictly need scaling, but Linear models DO.
    # We use make_pipeline to safely scale ONLY the training data.
    
    if problem_type == "regression":
        models = {
            "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        primary_metric = "R2 Score"
    else:
        models = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        primary_metric = "Accuracy"

    # 4. Train and Evaluate
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate Multiple Metrics
            metrics = {}
            if problem_type == "regression":
                score = r2_score(y_test, predictions)
                metrics["R2 Score"] = round(score, 4)
                metrics["MAE"] = round(mean_absolute_error(y_test, predictions), 4)
            else:
                score = accuracy_score(y_test, predictions)
                metrics["Accuracy"] = round(score, 4)
                # Weighted F1 handles multi-class imbalances better
                metrics["F1 Score"] = round(f1_score(y_test, predictions, average='weighted'), 4)
                
            # Add to results table
            metrics["Model"] = name
            results.append(metrics)
            
            # Track Winner
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name
                
        except Exception as e:
            results.append({"Model": name, primary_metric: "Failed", "Error": str(e)})

    # 5. Generate Feature Importance Plot
    plt.figure(figsize=(10, 6))
    
    importances = None
    
    # Handle extracting importances from Pipelines vs raw models
    final_estimator = best_model
    if hasattr(best_model, 'named_steps'): # It's a pipeline (Linear/Logistic)
        final_estimator = best_model.named_steps[list(best_model.named_steps.keys())[-1]]
    
    if hasattr(final_estimator, 'feature_importances_'):
        importances = final_estimator.feature_importances_
    elif hasattr(final_estimator, 'coef_'):
        importances = np.abs(final_estimator.coef_)
        if importances.ndim > 1: # Handle multi-class logistic regression
             importances = np.mean(importances, axis=0)

    if importances is not None:
        indices = np.argsort(importances)[-10:] # Top 10 features
        plt.barh(range(len(indices)), importances[indices], align='center', color='#4e79a7')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.title(f'Feature Importance ({best_model_name})')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, "Feature importance not available for this model", 
                 ha='center', va='center')

    # Convert results to DataFrame for nice display
    results_df = pd.DataFrame(results).sort_values(by=primary_metric, ascending=False)
    
    # 6. Construct Summary Message
    best_metrics = results_df.iloc[0].to_dict()
    metric_str = ", ".join([f"{k}={v}" for k, v in best_metrics.items() if k != "Model"])
    
    return results_df, f"Winner: {best_model_name}. Metrics: {metric_str}. (See plot for details)"