
<div align="center">


# Chess Master AI  <img src="./assets/img/chess.png" alt="chess icons" width="45"/> 

</div>

![Chess AI Banner](./assets/img/app.png)
<hr>

**Chess Master AI** is an intelligent chess assistant that leverages the power of Ollama, LangChain, and Streamlit to help users improve their chess skills. Utilizing Retrieval-Augmented Generation (RAG) on two comprehensive chess PDFs—containing the FIDE rules and a guide on how to play and excel at chess—this application provides users with valuable insights and personalized coaching.

## Features

- **Interactive User Interface**: Built with Streamlit for a seamless user experience.
- **Rule Retrieval**: Quickly access the official FIDE rules and guidelines for chess.
- **Personalized Coaching**: Get tailored advice on improving your chess skills using advanced models.
- **Embedded Knowledge**: Utilizes the `nomic-embed-text` for effective information retrieval from chess documents.
- **Advanced AI Models**: Integrates the latest `gemma2` model to enhance user learning.

## Requirements

Before running Chess Master AI, ensure you have the following installed:

1. **[Ollama](https://ollama.com/)**: An AI model management tool. Follow the instructions on their website to install Ollama on your system.

2. **Python**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

3. **Conda** (optional): If you prefer to use Conda, you can install it by following the instructions at [Anaconda](https://docs.anaconda.com/anaconda/install/).

## Setting Up Your Environment

### Using Python

1. **Create a Virtual Environment**:

   ```bash
   python -m venv ollama-chess-env
    ```
2. **Activate the Virtual Environment**:
    On Windows:
    ```bash
    ollama-chess-env\Scripts\activate
    ```
    On macOS/Linux:
    ```bash
    source ollama-chess-env/bin/activate
    ```

### Using Conda

1. **Create a New Conda Environment**:

   ```bash
   conda create --name ollama-chess-env python=3.11.10
    ```
2. **Activate the Conda Environment:**:
    ```bash
    conda activate ollama-chess-env
    ```
## Installation

Once your environment is set up, navigate to your project directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

After the setup is complete, you can start the Streamlit application:

```bash
streamlit run chessLLM.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

### How to Use Chess Master AI

- **Ask Questions**: Type your chess-related questions in the input box.
- **Get Suggestions**: Receive personalized suggestions on how to improve your gameplay.
- **Access Rules**: Retrieve specific rules and guidelines as needed.

## Contributions

Contributions are welcome! If you would like to contribute to the Chess Master AI project, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/YourFeatureName
