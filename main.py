from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Literal
from Agent.agent import DatasetValidationAgent
import tabula
import pdfplumber

app = FastAPI(title="Dataset Validation Agent", version="1.0.0")
agent = DatasetValidationAgent()

def load_tabular_data(file_content: bytes, filename: str) -> pd.DataFrame:
    """Load tabular data from various file formats."""
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'csv':
        return pd.read_csv(io.BytesIO(file_content))
    elif file_ext in ['xlsx', 'xls']:
        return pd.read_excel(io.BytesIO(file_content))
    elif file_ext == 'tsv':
        return pd.read_csv(io.BytesIO(file_content), sep='\t')
    elif file_ext == 'pdf':
        # Extract tables from PDF
        tables = tabula.read_pdf(io.BytesIO(file_content), pages='all')
        if tables:
            return tables[0]  # Return first table found
        else:
            raise HTTPException(status_code=400, detail="No tables found in PDF")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

@app.post("/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    target_type: Literal["categorical", "numeric"] = Form(...)
):
    """Validate uploaded tabular dataset."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Load as DataFrame
        df = load_tabular_data(file_content, file.filename)
        
        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in dataset")
        
        # Run validation
        report = agent.validate_dataset(df, target_column, target_type)
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "validation_report": report
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """API health check."""
    return {"message": "Dataset Validation Agent API", "status": "active"}

@app.get("/supported-formats")
async def supported_formats():
    """List supported file formats."""
    return {
        "supported_formats": [
            ".csv - Comma-separated values",
            ".xlsx/.xls - Excel files", 
            ".tsv - Tab-separated values",
            ".pdf - PDF with tables"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)