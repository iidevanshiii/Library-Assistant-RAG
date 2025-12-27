import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- 1. Load the Embedding Model ---
print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# --- 2. Initialize ChromaDB ---
db_client = chromadb.PersistentClient(path="./db")
collection_name = "library_assistant"
try:
    db_client.delete_collection(name=collection_name)
    print(f"Deleted old collection (if any): {collection_name}")
except:
    pass # Collection didn't exist, which is fine

collection = db_client.create_collection(name=collection_name)
print(f"Created new, empty collection: {collection_name}")

# --- 3. Define Your 5 Data Files ---
# Add all your new file names to this list
data_files = [
    "Harish data new.json",
    "Goutham data new.json",
    "Vaishnavi data new.json",
    "Devanshi data new.json",
    "Rakesh data new.json"
]

documents_to_add = []
metadata_to_add = []
ids_to_add = []
doc_id_counter = 1

# --- 4. Process All Files ---
for file_name in data_files:
    print(f"--- Processing {file_name} ---")
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            books = json.load(f)
    except FileNotFoundError:
        print(f"WARNING: '{file_name}' not found. Skipping.")
        continue
    except json.JSONDecodeError:
        print(f"WARNING: '{file_name}' is not a valid JSON. Skipping.")
        continue

    # Loop through each book in the current file
    for book in books:
        # Standardize key names (handles 'Book Title' or 'book_title')
        book_title = book.get('book_title', book.get('Book Title', 'Unknown Title'))
        
        # --- Chunk Type 1: The Book Summary ---
        book_summary = book.get('summary', '')
        
        # QUALITY CHECK: Only index if the summary is good (not empty, not "N/A")
        if book_summary and len(book_summary) > 20 and book_summary != "N/A":
            book_chunk_text = f"Book: {book_title}. Summary: {book_summary}"
            
            documents_to_add.append(book_chunk_text)
            metadata_to_add.append({"type": "book", "title": book_title})
            ids_to_add.append(f"book_{doc_id_counter}")
            doc_id_counter += 1
        
        # --- Chunk Type 2: The Chapter Summaries ---
        chapters = book.get('chapters', [])
        for chapter in chapters:
            # Standardize key names ('title' or 'chapter_name')
            chapter_name = chapter.get('chapter_name', chapter.get('title', 'Unknown Chapter'))
            chapter_summary = chapter.get('chapter_summary', chapter.get('summary', ''))
            
            # QUALITY CHECK: Only index if the summary is good!
            if chapter_summary and len(chapter_summary) > 20 and chapter_summary != "N/A" and "Abstract not found" not in chapter_summary:
                chapter_chunk_text = f"Book: {book_title}. Chapter: {chapter_name}. Summary: {chapter_summary}"
                
                documents_to_add.append(chapter_chunk_text)
                metadata_to_add.append({"type": "chapter", "title": book_title, "chapter": chapter_name})
                ids_to_add.append(f"chapter_{doc_id_counter}")
                doc_id_counter += 1
            else:
                # This helps you find bad data in your files
                print(f"Skipping empty/bad chapter: '{chapter_name}' in '{book_title}'")

print(f"\nCreated a total of {len(documents_to_add)} high-quality text chunks from all files.")

# --- 5. Embed and Store in ChromaDB ---
if documents_to_add:
    print("Embedding documents (this may take a moment)...")
    embeddings = model.encode(documents_to_add)

    print("Adding documents and embeddings to ChromaDB...")
    collection.add(
        embeddings=embeddings,
        documents=documents_to_add,
        metadatas=metadata_to_add,
        ids=ids_to_add
    )

    print("\n--- Indexing Complete ---")
    print(f"Total documents indexed in '{collection_name}': {collection.count()}")
else:
    print("No valid documents with summaries were found to index.")
    