from pydantic import BaseModel
from typing import List, Any, Dict, Optional

class DatasetPreview(BaseModel):
    filename: str
    content_type: str
    shape: List[int]  # [rows, columns]
    columns: List[str]
    dtypes: Dict[str, str] # e.g., {"age": "int64", "salary": "float64"}
    summary_stats: Dict[str, Any] # Basic describe() output
    first_rows: List[Dict[str, Any]] # JSON representation of .head()

class ResponseModel(BaseModel):
    message: str
    file_id: str # Useful for tracking the file in future steps
    preview: DatasetPreview
    description: Optional[str] = None

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    text_output: str
    image_output: Optional[str] = None # Base64 PNG string
    error: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    file_id: str # We need to know WHICH file to analyze

class ChatResponse(BaseModel):
    response_text: str
    generated_code: str
    image_output: Optional[str] = None

