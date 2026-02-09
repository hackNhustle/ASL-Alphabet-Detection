from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .predictor import ASLPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    try:
        app.state.predictor = ASLPredictor()
        logger.info("ASL predictor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ASL predictor: {e}")
        app.state.predictor = None
    yield

app = FastAPI(title="ASL Alphabets Service", lifespan=lifespan)

# Add CORS allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "asl-detection"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    predictor = getattr(app.state, "predictor", None)
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
