FROM huggingface/transformers-pytorch-gpu:latest
COPY ./ /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app"]
