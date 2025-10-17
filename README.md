# Sistema de Gamificação para Fisioterapia Infantil com IA

> Projeto desenvolvido para a disciplina de Inteligência Artificial II do Centro Universitário Mário Pontes Jucá (UMJ)

Este projeto é um sistema de gamificação em formato de aplicação desktop, desenvolvido com o objetivo de incentivar e engajar crianças em sessões de reabilitação motora, transformando o processo em uma experiência lúdica e motivadora. A aplicação utiliza visão computacional para analisar os movimentos do usuário em tempo real e oferece feedback interativo através de um personagem animado.

---

## 🚀 Funcionalidades Principais

* **Seleção de Múltiplos Exercícios:** A tela inicial permite escolher entre diferentes "missões", cada uma correspondendo a um exercício de reabilitação.
* **Detecção de Pose em Tempo Real:** Utilizando a biblioteca MediaPipe, o sistema captura 33 pontos corporais do usuário através da webcam para analisar a postura.
* **Classificação de Movimentos com IA:** Para cada exercício, um modelo de Machine Learning treinado com Scikit-learn classifica a pose do usuário em estágios (`down`, `middle`, `up`, etc.), validando a execução correta do movimento.
* **Interface Gamificada e Interativa:** Um super-herói animado reage ao desempenho do jogador. A cada repetição correta, o herói executa uma animação de comemoração e exibe mensagens de incentivo.
* **Sistema de Recompensa:** A interface inclui um contador de repetições, uma barra de progresso e um sistema de estrelas que são concedidas ao atingir metas, reforçando positivamente o esforço do usuário.
* **Feedback de Conclusão:** Ao final de cada missão, uma tela de vitória exibe as estrelas conquistadas e celebra o esforço da criança.

---

## 🛠️ Tecnologias e Arquitetura

O projeto foi desenvolvido inteiramente em **Python** e segue uma arquitetura modular de 3 etapas:

1.  **Coleta de Dados (`/coleta_dados`)**
    * Um script interativo (`coletar_dados_...py`) utiliza **OpenCV** para exibir frames de um vídeo de exemplo.
    * O **MediaPipe** é usado para extrair as 33 coordenadas do corpo (x, y, z, visibilidade) de cada frame.
    * O operador pode pausar, avançar e rotular cada frame com uma classe (`down`, `middle`, `up`, etc.), salvando os dados estruturados em um arquivo `.csv`.

2.  **Treinamento do Modelo (`/treinar_modelo`)**
    * Um script (`treinar_modelo_...py`) carrega os dados do `.csv` usando **Pandas**.
    * A biblioteca **Scikit-learn** é utilizada para treinar um modelo de classificação (Regressão Logística com `StandardScaler`).
    * O script avalia a performance do modelo e salva o objeto treinado em um arquivo `.pkl` usando **Pickle**.

3.  **Aplicação Principal (GUI)**
    * A interface gráfica foi construída com **Tkinter** e estilizada para ser amigável para crianças.
    * A aplicação carrega dinamicamente o modelo `.pkl` correspondente ao exercício escolhido.
    * Ela gerencia uma máquina de estados para validar a sequência de movimentos e contar as repetições.
    * A biblioteca **Pillow (PIL)** é usada para integrar os frames de vídeo do OpenCV e as animações do herói na interface Tkinter.

---

## ⚙️ Como Executar o Projeto

### Pré-requisitos
* Python 3.10+
* Uma webcam conectada

### Instalação
1.  Clone este repositório:
    ```bash
    git clone [https://github.com/Maycon-Fidelis/projeto_IA2.git](https://github.com/Maycon-Fidelis/projeto_IA2.git)
    cd seu-repositorio
    ```

2.  Crie e ative um ambiente virtual:
    ```bash
    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  Crie um arquivo `requeriments.txt` com o seguinte conteúdo:
    ```txt
    opencv-python
    mediapipe
    scikit-learn
    pandas
    numpy
    Pillow
    ```

4.  Instale as dependências:
    ```bash
    pip install -r requeriments.txt
    ```

### Execução
Para iniciar a aplicação, execute o script principal:
```bash
python testar_modelo.py
```
*(**Nota:** Renomeie `testar_modelo.py` para o nome do seu arquivo principal, se for diferente.)*

---

## 🧑‍💻 Equipe e Responsabilidades

| Cargo | Objetivo | Responsável |
| :--- | :--- | :--- |
| **PM / Front-end** | Responsável por organizar e entregar o projeto. Implementação do jogo e integração com os modelos. | Maycon Vinicius |
| **QA/Documentação** | Elaboração de testes, relatórios e entrega final. | Daniel Melo |
| **Data Engineer** | Coleta e organização dos dados (imagens, áudios e poses). | Fagner Lucas |
| **Modeller** | Treinamento e ajuste dos modelos. | Ademildo Costa |
