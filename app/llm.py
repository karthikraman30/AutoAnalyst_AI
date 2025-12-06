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
    """
    Sends the user's query + dataset metadata to Gemini 
    and asks for Python code in return.
    """
    
    # 1. Construct the "System Prompt"
    # We teach Gemini its role and constraints here.
    prompt = f"""
    You are an expert Python Data Scientist Assistant.
    
    CONTEXT:
    The user has uploaded a dataset. It is ALREADY loaded into a pandas DataFrame named 'df'.
    
    DATASET METADATA:
    - Columns: {columns}
    - Summary Statistics: {summary}
    
    USER REQUEST:
    "{query}"
    
    YOUR GOAL:
    Write Python code to answer the request.
    
    RULES:
    1. Use 'df' directly. DO NOT reload the file.
    2. If the user asks for a plot, use 'matplotlib.pyplot' or 'seaborn'.
    3. If the user asks for a number/text, print() it.
    4. RESPOND ONLY WITH CODE. No markdown, no explanations, no ```python``` wrappers.
    5. Handle potential errors (like missing values) gracefully if possible.
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