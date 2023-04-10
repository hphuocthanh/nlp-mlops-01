# nlp-mlops-01
An MLOPs project based on the blogging series: https://www.ravirajag.dev/blog

Project goals:
- Implement an end-to-end lifecycle of MLOPs

High-quality modeling is not the project's current focus

Things I've done:
- Create data module and model using PyTorch Lightning (2.0.0)
- Monitor and model visualization using Weight and Biases
  -- Plot confusion matrix after each validation epoch end, loss, accuracy..
- Fetch configs from hydra
  -- Parameters for training, model and processing
- Data Version Control with DVC
  -- Save model on GG Drive
- Model packaging with ONNX
  -- ONNX + ONNXRuntime help improve inference and allow a flexible way to deploy models
- Dockerize model
  -- Run the model on any platforms without worrying about dependencies or setting up environments
- Updating...

First look at dataloader which sets up and prepares data. Then, head to model to see how the model is created. `train.py` is where we train the model and call `visualization.py` to log metrics to wandb. Finally, `inference.py` or `inference_w_onnx.py` are where we create inference.

`app.py` is served with FastAPI to allow basic REST endpoints.
`e2e_model.ipynb` is basically an end-to-end walkthrough of preparing data and training the model and inference (without any logging and metrics)

The blogging series are quite old and I'm using the newest (atm) libraries so there are tweaks there and then.
