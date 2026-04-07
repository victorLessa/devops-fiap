from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import time

# 1. Inicialização da API
app = FastAPI(
    title="Caixa Sentiment Analysis API",
    description="API para análise de sentimento de feedbacks utilizando BERT Multilingual",
    version="1.0.0"
)

# 2. Carregamento do Modelo (Singleton Pattern)
print("🕒 Carregando modelo BERT... Isso pode levar alguns segundos.")
classifier = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
print("✅ Modelo carregado com sucesso!")

# 3. Definição do Esquema de Entrada (Input Validation)
class FeedbackRequest(BaseModel):
    text: str

# 4. Endpoints
@app.get("/")
def read_root():
    return {"message": "Caixa Sentiment API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "bert-base-multilingual-uncased"}

@app.post("/predict")
def predict_sentiment(request: FeedbackRequest):
    start_time = time.time()
    
    # Realiza a predição
    prediction = classifier(request.text)[0]
    
    latency = time.time() - start_time
    
    return {
        "text": request.text,
        "label": prediction['label'],
        "score": round(prediction['score'], 4),
        "latency_seconds": round(latency, 4)
    }
