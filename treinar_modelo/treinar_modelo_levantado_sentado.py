# arquivo: treinar_modelo_asas.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report # <<< Adicionado para um relatório detalhado
import pickle
import os

# --- CONFIGURAÇÕES ---
# <<< MUDANÇA 1: Aponta para o arquivo CSV com os 3 estágios >>>
INPUT_CSV_PATH = './coord_videos/asas_de_super_heroi.csv'

# <<< MUDANÇA 2: Define um novo caminho e nome para o modelo treinado >>>
OUTPUT_MODEL_PATH = './modelos/levantar_sentar.pkl'
# ---------------------

# Garante que a pasta 'modelos' exista antes de salvar
os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

df = pd.read_csv(INPUT_CSV_PATH)
print("Dados carregados com sucesso!")
print("Amostras por classe:")
print(df['class'].value_counts()) # Mostra quantas amostras você tem para 'down', 'middle' e 'up'

X = df.drop('class', axis=1)
y = df['class']

# Divide em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nIniciando treinamento com {len(X_train)} amostras...")

pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
}

# Treina o modelo (nenhuma mudança aqui, o scikit-learn lida com as 3 classes)
model = pipelines['lr']
model.fit(X_train, y_train)

print("Treinamento concluído!")

# Avalia o modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Acurácia do modelo nos dados de teste: {accuracy * 100:.2f}%")

# <<< MELHORIA: EXIBE UM RELATÓRIO DETALHADO POR CLASSE >>>
print("\n📊 Relatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))

# Salva o modelo treinado
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"🚀 Modelo salvo com sucesso em '{OUTPUT_MODEL_PATH}'")