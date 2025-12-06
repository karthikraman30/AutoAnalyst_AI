import pandas as pd
import os
import shutil
import uuid
from fastapi import UploadFile, HTTPException

UPLOAD_DIR = "temp_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_file_locally(file: UploadFile) -> str:
    """
    Saves the uploaded file with a unique name to avoid conflicts.
    Returns the file path.
    """
    # Generate unique ID to prevent filename collisions
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    new_filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_path, file_id

def read_dataset(file_path: str):
    """
    Helper to read CSV/Excel with error handling for encodings.
    Tries UTF-8 first, then Latin-1 (common for financial data), then CP1252.
    """
    if file_path.endswith('.csv'):
        # Try default UTF-8 first
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            # Fallback to Latin-1 (common for Excel-generated CSVs)
            try:
                return pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                # Last resort fallback
                return pd.read_csv(file_path, encoding='cp1252')
                
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_and_preview_data(file_path: str, original_filename: str, content_type: str):
    """
    Reads CSV/Excel using the robust reader and returns metadata.
    """
    try:
        # Use the helper function here so we don't crash on encoding errors
        df = read_dataset(file_path)

        preview = {
            "filename": original_filename,
            "content_type": content_type or 'application/octet-stream',
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            # Convert NaN to None for valid JSON
            "summary_stats": df.describe().to_dict(), 
            "first_rows": df.head().replace({float('nan'): None}).to_dict(orient='records')
        }
        return preview

    except Exception as e:
        # Log this error to your terminal so you can see what went wrong
        print(f"Error processing file: {e}") 
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")