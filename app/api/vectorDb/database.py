import faiss
import numpy as np
import os
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import json
import time
import uuid

from pypdf import PdfReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup for LLM Service 
import sys
_current_script_dir = os.path.dirname(os.path.realpath(__file__))
_app_dir = os.path.abspath(os.path.join(_current_script_dir, "..", "..")) 

if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

try:
    from services.llm_service import get_groq_client, get_llm_synthesis_with_citations
    LLM_SERVICE_AVAILABLE = True
    print("INFO (database.py): Successfully imported LLM service.")
except ImportError as e:
    print(f"WARNING (database.py): Could not import from services.llm_service. LLM features will be disabled. Details: {e}")
    def get_groq_client():
        print("DEBUG (database.py): get_groq_client (dummy) called.")
        return None
    def get_llm_synthesis_with_citations(client, user_query, retrieved_chunks, llm_model_name=""):
        print("DEBUG (database.py): get_llm_synthesis_with_citations (dummy) called.")
        return "LLM Service Not Available due to import error.", {}
    LLM_SERVICE_AVAILABLE = False


#Configuration
SCRIPT_DIR_FOR_PATHS = os.path.dirname(os.path.realpath(__file__))

# MODIFICATION
DEFAULT_DOCUMENTS_DIR = os.path.join(SCRIPT_DIR_FOR_PATHS, "data")

DEFAULT_FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR_FOR_PATHS, "document_index.faiss")
DEFAULT_METADATA_PATH = os.path.join(SCRIPT_DIR_FOR_PATHS, "document_metadata.json")
MODEL_NAME = 'all-MiniLM-L6-v2'

# Helper Functions for Document Processing
def extract_text_from_pdf_pypdf(pdf_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            print(f"WARN (extract_text_from_pdf_pypdf): No pages found in PDF: {pdf_path}")
            return ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e_page:
                print(f"WARN (extract_text_from_pdf_pypdf): Could not extract text from page {i+1} of {pdf_path}. Error: {e_page}")
        return text.strip()
    except Exception as e_file:
        print(f"ERROR (extract_text_from_pdf_pypdf): Failed to read or process PDF file {pdf_path}. Error: {e_file}")
        return ""

def extract_text_from_image(image_path: str) -> str:
    text_from_ocr = ""
    try:
        img = Image.open(image_path)
        text_from_ocr = pytesseract.image_to_string(img)
        return text_from_ocr.strip()
    except pytesseract.TesseractNotFoundError:
        print("ERROR (extract_text_from_image): Tesseract is not installed or not found in your PATH.")
        return ""
    except Exception as e:
        print(f"ERROR (extract_text_from_image): Failed to OCR image {image_path}. Error: {e}")
        return ""

def read_text_from_txt(txt_path: str) -> str:
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"ERROR (read_text_from_txt): Failed to read text file {txt_path}. Error: {e}")
        return ""

def get_embedding_model(model_name=MODEL_NAME):
    print(f"INFO (get_embedding_model): Loading sentence transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("INFO (get_embedding_model): Sentence transformer model loaded successfully.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR (get_embedding_model): Failed to load SentenceTransformer model '{model_name}'. Error: {e}")
        raise

def get_text_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False
    )

# Indexing and Updating Documents 
def process_and_update_index(docs_dir, embedding_model, text_splitter,
                             faiss_index_path, metadata_path,
                             current_index=None, current_metadata=None):
    print(f"INFO (process_and_update_index): Processing documents in '{docs_dir}' to build/update index at '{faiss_index_path}'.")
    if current_metadata is None or not isinstance(current_metadata, dict) or \
       "chunk_ids_list" not in current_metadata or "metadata_store" not in current_metadata:
        current_metadata = {"chunk_ids_list": [], "metadata_store": {}}

    new_chunks_text_this_run = []
    new_chunk_metadata_this_run = []
    all_chunk_ids_list = list(current_metadata.get("chunk_ids_list", []))
    full_metadata_store = dict(current_metadata.get("metadata_store", {}))
    next_faiss_id = current_index.ntotal if current_index and hasattr(current_index, 'ntotal') else 0

    doc_filepaths = []
    if not os.path.isdir(docs_dir):
        print(f"ERROR (process_and_update_index): Documents directory not found or is not a directory: {docs_dir}")
        return current_index, current_metadata

    for root, _, files in os.walk(docs_dir):
        for file_name in files:
            if not file_name.startswith('.'): 
                doc_filepaths.append(os.path.join(root, file_name))

    if not doc_filepaths:
        print(f"INFO (process_and_update_index): No documents found in '{docs_dir}' to process.")
        if current_index is not None: faiss.write_index(current_index, faiss_index_path)
        with open(metadata_path, 'w') as f: json.dump(current_metadata, f, indent=2)
        return current_index, current_metadata

    print(f"INFO (process_and_update_index): Found {len(doc_filepaths)} potential documents to process in '{docs_dir}'.")
    processed_files_this_run_count = 0
    for doc_path in doc_filepaths:
        raw_text, doc_type, doc_filename = "", "", os.path.basename(doc_path)
        if doc_path.lower().endswith(".pdf"): raw_text, doc_type = extract_text_from_pdf_pypdf(doc_path), "pdf"
        elif doc_path.lower().endswith((".png", ".jpg", ".jpeg")): raw_text, doc_type = extract_text_from_image(doc_path), "image_ocr"
        elif doc_path.lower().endswith(".txt"): raw_text, doc_type = read_text_from_txt(doc_path), "text"
        else:
            print(f"  WARN (process_and_update_index): Skipping unsupported file type: {doc_filename}")
            continue
        if not raw_text:
            print(f"  WARN (process_and_update_index): No text extracted from '{doc_filename}', skipping.")
            continue

        processed_files_this_run_count += 1
        print(f"  INFO (process_and_update_index): Extracted text from '{doc_filename}'. Splitting into chunks...")
        chunks = text_splitter.split_text(raw_text)
        print(f"  INFO (process_and_update_index): Document '{doc_filename}' split into {len(chunks)} chunks.")

        for i_chunk, chunk_text in enumerate(chunks):
            if not chunk_text.strip(): continue
            chunk_id = str(uuid.uuid4())
            new_chunks_text_this_run.append(chunk_text)
            chunk_meta_entry = {"text": chunk_text, "source_doc": doc_filename, "doc_path_at_indexing": doc_path, "doc_type": doc_type, "chunk_num_in_doc": i_chunk + 1}
            new_chunk_metadata_this_run.append((chunk_id, chunk_meta_entry))

    if not new_chunks_text_this_run:
        print("INFO (process_and_update_index): No new text chunks generated to add to the index.")
        if current_index is not None: faiss.write_index(current_index, faiss_index_path)
        with open(metadata_path, 'w') as f: json.dump(current_metadata, f, indent=2)
        return current_index, current_metadata

    print(f"INFO (process_and_update_index): Generating embeddings for {len(new_chunks_text_this_run)} new chunks...")
    new_embeddings = embedding_model.encode(new_chunks_text_this_run, convert_to_tensor=False, show_progress_bar=True)
    new_embeddings = np.array(new_embeddings).astype('float32')
    dimension = new_embeddings.shape[1]
    index_to_update = current_index
    if index_to_update is None:
        print(f"INFO (process_and_update_index): Creating new FAISS index (dimension: {dimension}).")
        base_index = faiss.IndexFlatL2(dimension)
        index_to_update = faiss.IndexIDMap(base_index)
    elif index_to_update.d != dimension:
        print(f"CRITICAL ERROR: FAISS index dimension mismatch ({index_to_update.d} vs {dimension}). Rebuild needed or ensure consistent embedding models.")
        return current_index, current_metadata

    new_sequential_faiss_ids = np.arange(next_faiss_id, next_faiss_id + len(new_chunks_text_this_run)).astype('int64')
    index_to_update.add_with_ids(new_embeddings, new_sequential_faiss_ids)
    for i, (generated_chunk_id, meta_entry) in enumerate(new_chunk_metadata_this_run):
        expected_faiss_id_for_this_chunk = next_faiss_id + i
        while len(all_chunk_ids_list) <= expected_faiss_id_for_this_chunk: all_chunk_ids_list.append(None)
        all_chunk_ids_list[expected_faiss_id_for_this_chunk] = generated_chunk_id
        full_metadata_store[generated_chunk_id] = meta_entry
    final_metadata_to_save = {"chunk_ids_list": all_chunk_ids_list, "metadata_store": full_metadata_store}
    faiss.write_index(index_to_update, faiss_index_path)
    with open(metadata_path, 'w') as f: json.dump(final_metadata_to_save, f, indent=2)
    print(f"INFO (process_and_update_index): Indexing complete. Total vectors: {index_to_update.ntotal}. Added {len(new_chunks_text_this_run)} new chunks from {processed_files_this_run_count} files.")
    return index_to_update, final_metadata_to_save

# Loading Index and Metadata
def load_existing_index_and_metadata(faiss_index_path, metadata_path):
    print(f"DEBUG (load_existing_index_and_metadata): Received faiss_index_path = {faiss_index_path}")
    print(f"DEBUG (load_existing_index_and_metadata): Received metadata_path = {metadata_path}")
    loaded_index, loaded_metadata = None, {"chunk_ids_list": [], "metadata_store": {}}
    if os.path.exists(faiss_index_path):
        try:
            print(f"INFO (load_existing_index_and_metadata): Loading FAISS index from '{faiss_index_path}'...")
            loaded_index = faiss.read_index(faiss_index_path)
            print(f"INFO (load_existing_index_and_metadata): FAISS index loaded. Index dimension: {loaded_index.d}, Total vectors: {loaded_index.ntotal}")
        except Exception as e:
            print(f"ERROR (load_existing_index_and_metadata): Failed to load FAISS index '{faiss_index_path}': {e}.")
            if os.path.exists(faiss_index_path):
                try: os.remove(faiss_index_path); print(f"INFO: Removed potentially corrupted FAISS index: {faiss_index_path}")
                except OSError as re: print(f"  WARN: Could not remove potentially corrupted index {faiss_index_path}: {re}")
    else:
        print(f"INFO (load_existing_index_and_metadata): FAISS index file not found at '{faiss_index_path}'.")

    if os.path.exists(metadata_path):
        try:
            print(f"INFO (load_existing_index_and_metadata): Loading metadata from '{metadata_path}'...")
            with open(metadata_path, 'r', encoding='utf-8') as f: temp_meta = json.load(f)
            if isinstance(temp_meta, dict) and \
               "chunk_ids_list" in temp_meta and isinstance(temp_meta["chunk_ids_list"], list) and \
               "metadata_store" in temp_meta and isinstance(temp_meta["metadata_store"], dict):
                loaded_metadata = temp_meta
                print(f"INFO (load_existing_index_and_metadata): Metadata loaded. Total chunk IDs: {len(loaded_metadata.get('chunk_ids_list', []))}, Store entries: {len(loaded_metadata.get('metadata_store', {}))}")
                if loaded_index and hasattr(loaded_index, 'ntotal') and len(loaded_metadata.get('chunk_ids_list', [])) != loaded_index.ntotal:
                    print(f"WARN: Mismatch FAISS index size ({loaded_index.ntotal}) vs metadata ({len(loaded_metadata.get('chunk_ids_list', []))}). Rebuild suggested.")
            else:
                print(f"WARN: Metadata file '{metadata_path}' format error. Initializing fresh.")
                if os.path.exists(metadata_path):
                    try: os.remove(metadata_path); print(f"INFO: Removed malformed metadata: {metadata_path}")
                    except OSError as re: print(f"  WARN: Could not remove malformed metadata {metadata_path}: {re}")
        except Exception as e:
            print(f"ERROR loading/parsing metadata '{metadata_path}': {e}. Initializing fresh.")
            if os.path.exists(metadata_path):
                try: os.remove(metadata_path); print(f"INFO: Removed corrupted metadata: {metadata_path}")
                except OSError as re: print(f"  WARN: Could not remove corrupted metadata {metadata_path}: {re}")
    else:
        print(f"INFO (load_existing_index_and_metadata): Metadata file not found at '{metadata_path}'.")
    return loaded_index, loaded_metadata

#  Querying (FAISS search)
def search_documents(query_text, index, metadata_payload, embedding_model, k=3):
    if index is None or (hasattr(index, 'ntotal') and index.ntotal == 0): return []
    if not hasattr(embedding_model, 'encode'): return []
    try:
        query_embedding = embedding_model.encode([query_text], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype('float32')
    except Exception: return []
    if query_embedding.shape[1] != index.d: return []
    try:
        distances, faiss_ids_retrieved = index.search(query_embedding, k)
    except Exception: return []
    results, chunk_ids_list, metadata_store = [], metadata_payload.get("chunk_ids_list", []), metadata_payload.get("metadata_store", {})
    for i_res in range(len(faiss_ids_retrieved[0])):
        faiss_id = faiss_ids_retrieved[0][i_res]
        if faiss_id == -1 or faiss_id >= len(chunk_ids_list): continue
        actual_chunk_id = chunk_ids_list[faiss_id]
        if actual_chunk_id is None: continue
        chunk_meta = metadata_store.get(actual_chunk_id)
        if chunk_meta:
            current_distance = float(distances[0][i_res])
            score = 1.0 / (1.0 + current_distance) if current_distance >= 0 else 0.0
            results.append({"score": score, "distance": current_distance, "chunk_id": actual_chunk_id, "faiss_id": int(faiss_id), **chunk_meta})
    return results

# Chatbot Service Logic 
class DocumentChatService:
    def __init__(self, documents_dir=DEFAULT_DOCUMENTS_DIR,
                 faiss_index_path=DEFAULT_FAISS_INDEX_PATH,
                 metadata_path=DEFAULT_METADATA_PATH,
                 embedding_model_name=MODEL_NAME,
                 rebuild_on_init=False,
                 chunk_size=500, chunk_overlap=50):
        print(f"INFO (DocumentChatService): Initializing with documents_dir='{documents_dir}'")
        self.documents_dir = documents_dir
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.documents_dir, exist_ok=True)

        self.embedding_model = get_embedding_model(self.embedding_model_name)
        self.text_splitter = get_text_splitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.groq_client = get_groq_client() if LLM_SERVICE_AVAILABLE else None

        self.faiss_index = None
        self.metadata = {"chunk_ids_list": [], "metadata_store": {}}

        if rebuild_on_init:
            print("INFO (DocumentChatService): rebuild_on_init is True. Deleting existing index/metadata.")
            self._delete_existing_index_files()
        else:
            self.faiss_index, self.metadata = load_existing_index_and_metadata(
                self.faiss_index_path, self.metadata_path
            )
        self.update_knowledge_base()
        if self.faiss_index and hasattr(self.faiss_index, 'ntotal'):
            print(f"INFO (DocumentChatService): Initialization complete. Index has {self.faiss_index.ntotal} vectors.")
        else:
            print("WARN (DocumentChatService): Initialization complete, but FAISS index is not available or empty.")

    def _delete_existing_index_files(self):
        for f_path in [self.faiss_index_path, self.metadata_path]:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"INFO (DocumentChatService): Removed existing file: '{f_path}'.")
                except OSError as e:
                    print(f"ERROR (DocumentChatService): Could not remove file '{f_path}': {e}")

    def update_knowledge_base(self):
        print("INFO (DocumentChatService): Updating knowledge base...")
        self.faiss_index, self.metadata = process_and_update_index(
            self.documents_dir, self.embedding_model, self.text_splitter,
            self.faiss_index_path, self.metadata_path,
            current_index=self.faiss_index, current_metadata=self.metadata
        )

    def process_query(self, user_query: str, k_retrieval: int = 7, llm_model_to_use: str = "llama3-8b-8192"):
        if not self.faiss_index or not self.metadata.get("chunk_ids_list"):
            return "Knowledge base not ready or empty.", {}
        retrieved_chunks = search_documents(user_query, self.faiss_index, self.metadata, self.embedding_model, k=k_retrieval)
        if not retrieved_chunks: return "No relevant information found in documents.", {}
        unique_contexts, seen_keys = [], set()
        for item in retrieved_chunks:
            key = (item.get('source_doc'), item.get('chunk_num_in_doc'))
            if key not in seen_keys and item.get('text','').strip():
                unique_contexts.append(item); seen_keys.add(key)
        if not unique_contexts: return "Found some information, but it was not distinct or was empty.", {}

        if self.groq_client and LLM_SERVICE_AVAILABLE:
            return get_llm_synthesis_with_citations(self.groq_client, user_query, unique_contexts, llm_model_name=llm_model_to_use)
        else:
            response_text = "LLM service is unavailable. Here are some relevant snippets:\n"
            doc_map = {}
            for i, chunk in enumerate(unique_contexts[:2]):
                ref_label = f"SnippetRef{i+1}"
                response_text += f"\n{ref_label}: \"{chunk.get('text', '')[:200].strip()}...\" (Source: {chunk.get('source_doc')})\n"
                doc_map[ref_label] = f"{chunk.get('source_doc')} (Chunk {chunk.get('chunk_num_in_doc')})"
            return response_text, doc_map

# Example of Chatbot Test
if __name__ == '__main__':
    print("--- Document Chat Service Test (Adhering to specified paths) ---")

    print(f"Service will use Documents Directory: '{DEFAULT_DOCUMENTS_DIR}'") 
    print(f"Service will use FAISS Index: '{DEFAULT_FAISS_INDEX_PATH}'")
    print(f"Service will use Metadata: '{DEFAULT_METADATA_PATH}'")
    print(f"Ensure your .env file is at: '{os.path.join(SCRIPT_DIR_FOR_PATHS, '.env')}' (for llm_service.py)")

    if not os.path.exists(DEFAULT_DOCUMENTS_DIR):
        print(f"WARNING: The documents directory '{DEFAULT_DOCUMENTS_DIR}' does not exist. Creating it now.")
        os.makedirs(DEFAULT_DOCUMENTS_DIR, exist_ok=True)
    print(f"Please add your sample documents (PDFs, TXTs) to '{DEFAULT_DOCUMENTS_DIR}'.")


    should_rebuild_index_for_test = True 
                                         

    chat_service_instance = DocumentChatService(
        documents_dir=DEFAULT_DOCUMENTS_DIR,
        faiss_index_path=DEFAULT_FAISS_INDEX_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
        rebuild_on_init=should_rebuild_index_for_test
    )

    if chat_service_instance.faiss_index is None or \
       not hasattr(chat_service_instance.faiss_index, 'ntotal') or \
       not chat_service_instance.metadata.get("chunk_ids_list"):
        print("\nCRITICAL: Test chat service could not initialize its knowledge base properly.")
        if not os.listdir(DEFAULT_DOCUMENTS_DIR):
            print(f"HINT: The documents directory '{DEFAULT_DOCUMENTS_DIR}' is empty. Add documents and try again (with rebuild_on_init=True if needed).")
    else:
        print(f"\n--- Interactive Chat Test (using DocumentChatService instance) ---")
        print(f"Knowledge base has {chat_service_instance.faiss_index.ntotal} indexed chunks.")
        print("Type your query and press Enter. Type 'quit' or 'exit' to stop.")

        while True:
            try:
                user_input_query = input("You: ").strip()
                if user_input_query.lower() in ['quit', 'exit']: break
                if not user_input_query: continue

                chatbot_response, source_mapping = chat_service_instance.process_query(user_input_query)
                print("\nChatbot:")
                print(chatbot_response)
                if source_mapping:
                    print("\n--- Document Reference Key ---")
                    for ref, src in source_mapping.items(): print(f"  {ref}: {src}")
                print("-" * 70 + "\n")
            except KeyboardInterrupt: print("\nExiting chat test..."); break
            except Exception as e_loop:
                print(f"ERROR in interactive chat loop: {e_loop}")
                import traceback
                traceback.print_exc()

    print("\n--- Document Chat Service Test Finished ---")