from model import ColaModel
from dataloader import DataModule
from utils import timing
from scipy.special import softmax
import onnxruntime as ort
import numpy as np

class ColaONNXPredictor:
    def __init__(self, model_path):
        self.processor = DataModule()
        self.ort_session = ort.InferenceSession(model_path)
        self.labels = ["unacceptable", "acceptable"]

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        ort_outs = self.ort_session.run(None, ort_inputs)
        scores = softmax(ort_outs[0])[0]

        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "The boy is doing a standup comedy"
    predictor = ColaONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))

    sentences = ["The boy is doing a standup comedy"] * 10
    for sentence in sentences:
        predictor.predict(sentence)
