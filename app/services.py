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

def load_and_preview_data(file_path: str, original_filename: str, content_type: str):
    """
    Reads CSV/Excel, handles errors, and returns metadata.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            content_type = content_type or 'text/csv'  # Ensure content_type is not None
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            content_type = content_type or 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            raise ValueError("Unsupported file format")

        # Basic Pre-computation for Phase 1
        preview = {
            "filename": original_filename,
            "content_type": content_type or 'application/octet-stream',  # Fallback content type
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            # Convert NaN to None for valid JSON
            "summary_stats": df.describe().to_dict(), 
            "first_rows": df.head().replace({float('nan'): None}).to_dict(orient='records')
        }
        return preview

    except Exception as e:
        # In a real app, log this error
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")