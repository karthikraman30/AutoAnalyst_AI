import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configure the API Key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running the application.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # List available models for debugging
    available_models = [m.name for m in genai.list_models()]
    print("\n=== Available Gemini Models ===")
    for model in available_models:
        print(f"- {model}")
    print("===========================\n")
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    raise

def generate_code_from_query(query: str, columns: list, summary: dict) -> str:
    prompt = f"""
    You are an expert Python Data Scientist Assistant.
    
    CONTEXT:
    The user has uploaded a dataset. It is ALREADY loaded into a pandas DataFrame named 'df'.
    
    AVAILABLE POWER TOOLS (Use these preferentially):
    1. `issues = identify_issues(df)` -> Returns dictionary of missing values/duplicates.
    2. `df, log = auto_clean(df)` -> Automatically fills missing values and drops duplicates.
    3. `df, log = auto_encode(df)` -> Encodes text columns to numbers (REQUIRED before ML).
    4. `results, msg = find_best_model(df, target_col='Price')` -> Trains models and returns a comparison table.
    
    DATASET METADATA:
    - Columns: {columns}
    - Summary Statistics: {summary}
    
    USER REQUEST:
    "{query}"
    
    YOUR GOAL:
    Write Python code to answer the request.
    
    RULES:
    1. Use 'df' directly.
    2. If the user asks to "clean data", use `auto_clean`.
    3. If the user asks to "predict [column]" or "run ML", you MUST:
       a) Run `df, _ = auto_encode(df)`
       b) Run `results, msg = find_best_model(df, target_col='...')`
       c) Print the `results` and `msg`.
    4. ALWAYS print the output variables so the user can see them.
    5. RESPOND ONLY WITH CODE.
    """

    # 2. Call Gemini
    try:
        # Use the specified Gemini 2.5 Flash model
        model_name = 'gemini-2.5-flash'
        print(f"Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        if not response or not hasattr(response, 'text'):
            raise Exception("No valid response received from the model")
            
        # 3. Clean the output
        # Gemini might still wrap code in ```python ... ```. We strip that.
        code = response.text.replace("```python", "").replace("```", "").strip()
        
        return code
        
    except Exception as e:
        print("\n=== Gemini API Error ===")
        print(f"Error: {str(e)}")
        print("Available models:")
        try:
            for m in genai.list_models():
                print(f"- {m.name}")
        except Exception as list_error:
            print(f"Could not list models: {list_error}")
        print("======================\n")
        raise

def analyze_dataset(columns: list, summary: dict, first_rows: list) -> str:
    """
    Analyzes the uploaded data and generates a 'Welcome Message' 
    suggesting what the AutoAnalyst can do.
    """
    prompt = f"""
    You are an expert Data Scientist Assistant named AutoAnalyst. 
    A user just uploaded a new dataset. Analyze it and welcome them.
    
    DATASET METADATA:
    - Columns: {columns}
    - First 5 Rows: {first_rows}
    - Summary Stats: {summary}
    
    YOUR GOAL:
    Generate a friendly chat response that:
    1. **Briefly explains** what this data looks like (e.g., "This looks like a customer churn dataset...").
    2. **Suggests 3 concrete actions** the user can take using this project's capabilities (Plotting, Cleaning, or Machine Learning).
    
    FORMAT THE RESPONSE LIKE THIS:
    "Hello! I've loaded your dataset. 📂 
    
    It appears to contain data about [Topic] with columns like {columns[:3]}...
    
    Here is what I can do for you:
    1. 📊 **Visualize**: "Plot the distribution of [Column Name]"
    2. 🧹 **Clean**: "Check for missing values and clean the data"
    3. 🤖 **Predict**: "Train a model to predict [Target Column]"
    
    What would you like to start with?"
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I've loaded your data, but I couldn't generate an analysis. Error: {e}"