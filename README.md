# Rag with Deepseek and ChromaDB


This project implements a **Retrieval-Augmented Generation (RAG)** system using **DeepSeek LLM (R1)** via **Ollama**, enabling local LLM-powered question answering over your own documents.

### 📂 Project Structure

---

## 🚀 Features

- 🔗 Retrieval-Augmented Generation using **DeepSeek R1**
- 🧠 Local LLM processing via [Ollama](https://ollama.com/)
- 📄 Indexes and answers questions from documents in the `documents/` folder
- ✅ Simple CLI-based interface

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/cnrbd/deepseek_rag.git
cd deepseek_rag
```

### 2. Create virtual environment if you wanna
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the requirements
```bash
pip install -r requirements.txt
```

### 4. Start Ollama and pull the model (YOU NEED TO HAVE OLLAMA RUNNING IN THE BACK)
```bash
ollama pull deepseek:latest
```

### Now you can add documents to the documents folder and run app.py

### Finally, run the chatbot
```bash
python3 chat.py
```


