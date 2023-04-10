from fastapi import FastAPI
from inference_w_onnx import ColaONNXPredictor

app = FastAPI(title="NLP MLOps 01")


@app.get("/")
async def home():
    return "<h1>NLP Project</h1>"


# load the model
predictor = ColaONNXPredictor("./models/model.onnx")


@app.get("/predict")
async def get_prediction(text: str):
    result = predictor.predict(text)
    return result
