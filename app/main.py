from fastapi import FastAPI, File, UploadFile
from app.services import save_file_locally, load_and_preview_data
from app.executor import session_executor
from app.schemas import ResponseModel, CodeRequest, CodeResponse
import pandas as pd # Needed to load data into the session
import os

app = FastAPI(title="Data Scientist Assistant Backend")

@app.post("/upload", response_model=ResponseModel)
async def upload_dataset(file: UploadFile = File(...)):
    # 1. Save the file
    file_path, file_id = save_file_locally(file)
    
    # 2. Process using Pandas
    try:
        # Get content type from the uploaded file
        content_type = file.content_type
        
        # Load the data into a DataFrame
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            content_type = content_type or 'text/csv'
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            content_type = content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            raise ValueError("Unsupported file format")
        
        # Store the DataFrame in the executor's globals
        session_executor.globals['df'] = df
        
        # Generate preview data
        preview_data = load_and_preview_data(file_path, file.filename, content_type)
        
        return {
            "message": "File uploaded and processed successfully",
            "file_id": file_id,
            "preview": preview_data
        }
    except Exception as e:
        # Clean up if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e



@app.post("/upload", response_model=ResponseModel)
async def upload_dataset(file: UploadFile = File(...)):
    file_path, file_id = save_file_locally(file)
    
    # Existing logic...
    preview_data = load_and_preview_data(file_path, file.filename, file.content_type)

    # --- NEW PHASE 2 LOGIC ---
    # Automatically load the dataframe into the execution session as 'df'
    # This enables the user to immediately write code like "print(df.head())"
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    
    session_executor.locals['df'] = df
    # -------------------------
    
    return {
        "message": "File uploaded and loaded into Python session", # Updated message
        "file_id": file_id,
        "preview": preview_data
    }

@app.post("/execute", response_model=CodeResponse)
async def execute_python(request: CodeRequest):
    """
    Executes Python code sent by the user (or LLM).
    Has access to 'df' from the uploaded file.
    """
    result = session_executor.execute_code(request.code)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)