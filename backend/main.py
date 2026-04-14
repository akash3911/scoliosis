
from io import BytesIO

import cv2 as cv
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from scoliotect.yolov8_detector import predict_image_to_api_format


app = FastAPI(title="Scoliotect Minimal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"status": "ok", "predict_endpoint": "/v1/getprediction"}


async def _upload_to_bgr_image(image: UploadFile):
    pil_img = Image.open(BytesIO(await image.read())).convert("RGB")
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)


@app.post("/v1/getprediction")
async def get_prediction_v1(image: UploadFile = File(...)):
    image = await _upload_to_bgr_image(image)
    return predict_image_to_api_format(image)