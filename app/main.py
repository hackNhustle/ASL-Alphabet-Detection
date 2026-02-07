from fastapi import FastAPI, File, UploadFile, HTTPException
from .predictor import ASLPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASL Alphabets Service")

# Initialize predictor
try:
    predictor = ASLPredictor()
    logger.info("ASL predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ASL predictor: {e}")
    predictor = None

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "asl-detection"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        result, error = predictor.predict(content)
        
        if error:
            return {"error": error}
            
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
