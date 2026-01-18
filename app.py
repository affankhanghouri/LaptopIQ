from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional
import numpy as np

from src.constants_component import APP_HOST, APP_PORT
from src.pipeline_component.prediction_pipeline import LaptopData, LaptopPredictor
from src.pipeline_component.training_pipeline import TrainingPipeline
from src.logging_component import logger
from src.exception_component import MyException
import sys

# ==========================================================
# FastAPI App Initialization
# ==========================================================
app = FastAPI(title="Laptop Price Prediction API")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# ==========================================================
# CORS Configuration
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Home Page - Display Form
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the laptop price prediction form
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None, "error": None}
    )


# ==========================================================
# Training Endpoint
# ==========================================================
@app.get("/train")
async def train_route():
    """
    Trigger the ML training pipeline
    """
    try:
        logger.info("Starting training pipeline...")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        logger.info("Training completed successfully")
        return Response("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return Response(f"❌ Training failed: {str(e)}")


# ==========================================================
# Prediction Endpoint
# ==========================================================
@app.post("/predict", response_class=HTMLResponse)
async def predict_route(
    request: Request,
    Company: str = Form(...),
    TypeName: str = Form(...),
    Inches: float = Form(...),
    ScreenResolution: str = Form(...),
    Cpu: str = Form(...),
    Ram: str = Form(...),
    Memory: str = Form(...),
    Gpu: str = Form(...),
    OpSys: str = Form(...),
    Weight: str = Form(...)
):
    """
    Handle prediction request from the form
    """
    try:
        logger.info("Prediction request received")
        logger.info(f"Input data - Company: {Company}, TypeName: {TypeName}, Inches: {Inches}")

        # Ensure Ram has "GB" suffix
        if 'GB' not in str(Ram).upper():
            Ram = f"{Ram}GB"
        
        # Ensure Weight has "kg" suffix
        if 'kg' not in str(Weight).lower():
            Weight = f"{Weight}kg"

        # Create LaptopData object
        laptop_data = LaptopData(
            Company=Company,
            TypeName=TypeName,
            Inches=Inches,
            ScreenResolution=ScreenResolution,
            Cpu=Cpu,
            Ram=Ram,
            Memory=Memory,
            Gpu=Gpu,
            OpSys=OpSys,
            Weight=Weight
        )

        # Get DataFrame
        laptop_df = laptop_data.get_input_data_frame()
        logger.info(f"Input DataFrame created: {laptop_df.shape}")

        # Make prediction
        predictor = LaptopPredictor()
        prediction = predictor.predict(dataframe=laptop_df)
        
        predicted_price = round(float(prediction[0]), 2)
        logger.info(f"Prediction successful: ${predicted_price}")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": predicted_price,
                "error": None,
                # Pass back form values to keep them filled
                "form_data": {
                    "Company": Company,
                    "TypeName": TypeName,
                    "Inches": Inches,
                    "ScreenResolution": ScreenResolution,
                    "Cpu": Cpu,
                    "Ram": Ram.replace("GB", ""),
                    "Memory": Memory,
                    "Gpu": Gpu,
                    "OpSys": OpSys,
                    "Weight": Weight.replace("kg", "")
                }
            }
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        error_message = str(e)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction": None,
                "error": f"Prediction failed: {error_message}",
                "form_data": {}
            }
        )


# ==========================================================
# Health Check Endpoint
# ==========================================================
@app.get("/health")
async def health_check():
    """
    Check if the API is running
    """
    return {"status": "healthy", "message": "Laptop Price Predictor API is running"}


# ==========================================================
# App Runner
# ==========================================================
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)