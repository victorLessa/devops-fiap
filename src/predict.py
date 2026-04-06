import time
from transformers import pipeline

print("🚀 Iniciando carregamento do modelo...")
start_load = time.time()

# Carregando o modelo multilíngue (BERT)
classifier = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

end_load = time.time()
print(f"✅ Modelo carregado em {end_load - start_load:.2f} segundos.\n")

def predict(text):
    start_inf = time.time()
    result = classifier(text)[0]
    end_inf = time.time()
    
    print(f"Input: {text}")
    print(f"Label: {result['label']} | Confiança: {result['score']:.4f}")
    print(f"Tempo de inferência: {end_inf - start_inf:.4f} segundos\n")

if __name__ == "__main__":
    # Teste de validação
    predict("O atendimento no caixa eletrônico da Caixa foi excelente!")
    predict("Estou com dificuldades para acessar o Internet Banking.")
