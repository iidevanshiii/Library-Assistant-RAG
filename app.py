import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# --- 1. System Setup (Run once) ---

@st.cache_resource
def load_resources():
    """Load embedding model and connect to DB."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")
    
    print("Connecting to ChromaDB...")
    db_client = chromadb.PersistentClient(path="./db")
    collection = db_client.get_collection(name="library_assistant")
    print("Connected to collection.")
    return model, collection

# Load all resources
embedding_model, collection = load_resources()

# Set up Ollama client
# Make sure Ollama application is running!
try:
    ollama_client = ollama.Client()
    ollama_client.list() # Test connection
    print("Ollama connection successful.")
except Exception as e:
    print(f"Ollama connection failed: {e}")
    print("Please make sure the Ollama application is running.")
    st.error("Ollama connection failed. Please make sure the Ollama application is running.")
    st.stop()


# --- 2. RAG Functions ---

def retrieve_context(query_text, k=3):
    """Retrieve top-k context chunks from ChromaDB."""
    query_embedding = embedding_model.encode([query_text])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results['documents'][0]

def generate_response(query, context_chunks):
    """Format the prompt and get a response from Ollama."""
    
    # Format the context
    context_str = "\n\n".join(context_chunks)
    
    # The prompt template
    prompt_template = f"""
    You are a helpful library assistant.
    Answer the user's question based *only* on the following context:

    ---CONTEXT---
    {context_str}
    ---END CONTEXT---

    Question: {query}
    
    Answer:
    """
    
    # Send to Ollama
    # Make sure you have 'llama3' or change the model name
    try:
        response = ollama.chat(
            model='llama3', # <-- Change this if you pulled a different model
            messages=[{'role': 'user', 'content': prompt_template}],
            stream=False
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error with Ollama: {e}"

# --- 3. Streamlit Chat UI ---

st.title("📚 Library Assistant RAG App")
st.caption("Ask me about the books in our database!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Run the RAG pipeline ---
    with st.spinner("Thinking..."):
        
        # 1. Retrieve
        context = retrieve_context(prompt, k=3)
        
        # 2. Generate
        response_text = generate_response(prompt, context)
        
        # --- This is where your bad data will show up ---
        # If the context is wrong (like the Skinner data),
        # Ollama will *still* answer, but its answer will
        # be based on that wrong data.
        #
        
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})