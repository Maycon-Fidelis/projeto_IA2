# arquivo: treinar_modelo.py (com relatório de classificação)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# (# <<< MUDANÇA 1: IMPORTA A FUNÇÃO classification_report >>>)
from sklearn.metrics import accuracy_score, classification_report 
import pickle
import os

# --- CONFIGURAÇÕES ---
INPUT_CSV_PATH = '../coord_videos/alcancar_as_estrelas.csv'
OUTPUT_MODEL_PATH = '../modelos/alcancar_as_estrelas.pkl'
# ---------------------

# Garante que a pasta 'modelos' exista antes de salvar
os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

df = pd.read_csv(INPUT_CSV_PATH)

X = df.drop('class', axis=1)
y = df['class']

# Divide em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
}

# Treina o modelo
model = pipelines['lr']
model.fit(X_train, y_train)

# Avalia o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Acurácia Geral do Modelo: {accuracy * 100:.2f}%")

# (# <<< MUDANÇA 2: GERA E EXIBE O RELATÓRIO DE CLASSIFICAÇÃO >>>)
print("\n📊 Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))

# Salva o modelo treinado
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\n🚀 Modelo salvo com sucesso em '{OUTPUT_MODEL_PATH}'")