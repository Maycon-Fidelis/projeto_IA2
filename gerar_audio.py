import os
from gtts import gTTS

# 1. Defina o texto e o nome do arquivo para cada som
AUDIO_TEXTS = {
    "success": "Repetição concluída! Ótimo!",
    "transition": "Correto!",
    "start": "Missão iniciada! Boa sorte!",
    "complete": "Parabéns! Missão concluída com sucesso!",
    "error": "Houve um equivoco ao fazer a atividade!"
}

# 2. Configure o diretório de saída
OUTPUT_DIR = "assets/audio"

# Crie o diretório de saída se ele não existir
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Diretório criado: {OUTPUT_DIR}")

# 3. Itere e gere os arquivos MP3
print("\nIniciando geração de áudios...")

for filename, text in AUDIO_TEXTS.items():
    # Nome do arquivo de saída
    output_path = os.path.join(OUTPUT_DIR, f"{filename}.mp3")
    
    try:
        # Crie o objeto gTTS
        tts = gTTS(text=text, lang='pt', tld='com', slow=False)
        
        # Salve o arquivo MP3
        tts.save(output_path)
        print(f"Arquivo gerado: {output_path}")
        
    except Exception as e:
        print(f"ERRO ao gerar o áudio {filename}: {e}")
        print("Verifique sua conexão com a internet, pois gTTS precisa dela.")

print("\nGeração de áudios concluída.")