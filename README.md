# 📚 Library Assistant RAG

A fully local **Retrieval-Augmented Generation (RAG)** application built using **Streamlit**, **ChromaDB**, **Sentence Transformers**, and **Ollama**.  
The system answers user queries strictly based on a curated library dataset containing book and chapter summaries.

---

## 🚀 Features
- Local semantic search using ChromaDB
- Embeddings with `all-MiniLM-L6-v2`
- Local LLM inference using Ollama (`llama3`)
- Streamlit-based chat interface
- Fully offline after setup (no cloud APIs)

---

## 🗂️ Project Structure

```
Library-Assistant-RAG/
│
├── app.py                  # Streamlit RAG application
├── build_index.py          # Script to create embeddings & vector index
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and folders
├── README.md               # Project documentation
│
├── data/                   # Library dataset (JSON files)
│   ├── Statistics Data.json
│   ├── Machine Learning and Statistics Data.json
│   ├── Management Data.json
│   ├── Political Science Data.json
│   └── Computer Science Data.json
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/iidevanshiii/Library-Assistant-RAG.git
cd Library-Assistant-RAG
```

### 2️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Install and Run Ollama

Download Ollama from:
https://ollama.com

Pull the required model:
```bash
ollama pull llama3
```

Ensure Ollama is running in the background before starting the app.

---

## 🧩 Build the Vector Index

Run this once to generate embeddings and store them in ChromaDB:
```bash
python build_index.py
```

---

## 💬 Run the Application

```bash
streamlit run app.py
```

---

## 📌 Notes
- The `db/` folder is auto-generated and should not be committed.
- Answers are generated **only from retrieved context** (true RAG behavior).
- Works completely offline after initial setup.

---

## 🎯 Use Cases
- Library or academic search assistant
- NLP / RAG learning project
- Capstone project

---

## 👩‍💻 Author
**Devanshi Sharma**

---

## 📄 License
MIT License
