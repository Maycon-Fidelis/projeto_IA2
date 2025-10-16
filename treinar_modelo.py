# arquivo: treinar_modelo.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# O INPUT_CSV_PATH deve apontar para o arquivo gerado pelo coletor de dados.
# O OUTPUT_MODEL_PATH define o nome do arquivo do modelo treinado (.pkl) que será gerado.
# --- CONFIGURAÇÕES ---
INPUT_CSV_PATH = 'coords_treino.csv'
OUTPUT_MODEL_PATH = 'pose_classifier_model.pkl'
# ---------------------

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
print(f"Acurácia do modelo nos dados de teste: {accuracy * 100:.2f}%")

with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo salvo com sucesso em '{OUTPUT_MODEL_PATH}'")