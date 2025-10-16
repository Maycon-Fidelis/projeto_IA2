import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflowjs as tfjs # A biblioteca que faz a "tradução"
import os

# --- CONFIGURAÇÃO ---
DATASET_PATH = "coords_treino.csv"
MODEL_OUTPUT_FOLDER = "web_model" # Nome da pasta de saída
# --------------------

# 1. Carrega e prepara os dados
df = pd.read_csv(DATASET_PATH)
X = df.drop('class', axis=1)
y = df['class']

# Converte rótulos de texto ('up', 'down') para números (1, 0)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. Cria o modelo de rede neural com Keras (a "receita em português")
# CÓDIGO CORRIGIDO E MAIS EXPLÍCITO
model = tf.keras.Sequential([
    # Adicionamos uma camada de entrada dedicada e explícita
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    
    # A primeira camada Dense agora não precisa mais do argumento 'input_shape'
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Treina o modelo
print("\nIniciando treinamento do modelo...")
model.fit(X_train, y_train, epochs=50)

# 4. AVALIAÇÃO (Opcional, mas recomendado)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Acurácia do modelo: {accuracy * 100:.2f}%")

# 5. A CONVERSÃO (O PASSO MAIS IMPORTANTE)
# Cria a pasta de saída se ela não existir
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)
# Usa o conversor para "traduzir" o modelo
tfjs.converters.save_keras_model(model, MODEL_OUTPUT_FOLDER)

print(f"\n✅ Modelo 'traduzido' e salvo na pasta '{MODEL_OUTPUT_FOLDER}'")