from fastapi import FastAPI, File, UploadFile, HTTPException
# Import necessary services and schemas
from app.services import save_file_locally, load_and_preview_data, read_dataset
from app.executor import session_executor
from app.llm import generate_code_from_query, analyze_dataset
from app.schemas import ResponseModel, CodeRequest, CodeResponse, ChatRequest, ChatResponse
import os
import uvicorn

app = FastAPI(title="Data Scientist Assistant Backend")

# Global dictionary to store metadata in memory
# This acts as a simple "Brain Memory" so the LLM knows what columns exist.
METADATA_STORE = {} 

@app.post("/upload", response_model=ResponseModel)
async def upload_dataset(file: UploadFile = File(...)):
    file_path, file_id = save_file_locally(file)
    
    try:
        # 1. Generate Metadata
        preview_data = load_and_preview_data(file_path, file.filename, file.content_type)
        
        # 2. Load into Session
        df = read_dataset(file_path)
        session_executor.locals['df'] = df
        
        # 3. NEW: Generate the Chat Explanation
        ai_welcome_message = analyze_dataset(
            preview_data['columns'],
            preview_data['summary_stats'],
            preview_data['first_rows']
        )
        
        # 4. Save Metadata
        METADATA_STORE[file_id] = {
            "columns": preview_data['columns'],
            "summary": preview_data['summary_stats']
        }
        
        return {
            "message": "File uploaded",
            "file_id": file_id,
            "preview": preview_data,
            "description": ai_welcome_message # <--- Send it back
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

@app.post("/execute", response_model=CodeResponse)
async def execute_python(request: CodeRequest):
    """
    Directly executes Python code.
    Used by the LLM (or for testing purposes).
    """
    result = session_executor.execute_code(request.code)
    return result

@app.post("/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest):
    """
    The Intelligent Layer:
    1. Receives user question (e.g., "Plot salary distribution")
    2. Uses Gemini to write the Python code.
    3. Executes the code.
    4. Returns the result (text + image).
    """
    
    # 1. Retrieve Metadata
    if request.file_id not in METADATA_STORE:
        raise HTTPException(status_code=404, detail="File metadata not found. Please upload file first.")
        
    metadata = METADATA_STORE[request.file_id]
    
    # 2. Get Python Code from Gemini
    try:
        generated_code = generate_code_from_query(
            query=request.message,
            columns=metadata['columns'],
            summary=metadata['summary']
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    # 3. Execute the Code
    execution_result = session_executor.execute_code(generated_code)
    
    # 4. Handle Execution Errors (if the AI wrote bad code)
    if execution_result['error']:
        return {
            "response_text": f"I tried to run the code, but ran into an error:\n{execution_result['error']}",
            "generated_code": generated_code,
            "image_output": None
        }
        
    # 5. Return Success
    return {
        "response_text": execution_result['text_output'] or "Done! (Check the plot)",
        "generated_code": generated_code,
        "image_output": execution_result['image_output']
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)