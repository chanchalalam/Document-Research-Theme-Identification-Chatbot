import os
import sys
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import traceback

_main_py_dir = os.path.dirname(os.path.realpath(__file__)) 
_backend_dir = os.path.dirname(_main_py_dir) 

if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)  
# Core service and default path import
from app.api.vectorDb.database import (
    DocumentChatService,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_FAISS_INDEX_PATH,
    DEFAULT_METADATA_PATH
)
SERVICE_IMPORTS_OK = True
print("INFO (main.py - Flask): Successfully imported DocumentChatService and path defaults.")

app = Flask(__name__, template_folder='templates')

app.secret_key = os.urandom(24) 

doc_chat_service_instance: DocumentChatService | None = None
services_are_initialized_successfully = False

# Initializes the core document chat service
def initialize_core_services(rebuild=False):
    global doc_chat_service_instance, services_are_initialized_successfully
    if not SERVICE_IMPORTS_OK:
        print("FLASK_APP_INIT: Cannot initialize core services because imports failed.")
        services_are_initialized_successfully = False
        doc_chat_service_instance = DocumentChatService() 
        return

    print(f"FLASK_APP_INIT: Attempting to initialize DocumentChatService (rebuild={rebuild})...")
    try:
        # Attempt to create main DocumentChatService
        doc_chat_service_instance = DocumentChatService(
            documents_dir=DEFAULT_DOCUMENTS_DIR,
            faiss_index_path=DEFAULT_FAISS_INDEX_PATH,
            metadata_path=DEFAULT_METADATA_PATH,
            rebuild_on_init=rebuild 
        )
        # Check for successful initialization & DocumentChatService logs
        if doc_chat_service_instance and doc_chat_service_instance.faiss_index is not None and doc_chat_service_instance.groq_client is not None:
            print(f"FLASK_APP_INIT: DocumentChatService initialized. Index vectors: {getattr(doc_chat_service_instance.faiss_index, 'ntotal', 'N/A')}")
            services_are_initialized_successfully = True
        else:
            print("FLASK_APP_INIT: DocumentChatService initialized with ISSUES (FAISS index or Groq client might be unavailable). Check DocumentChatService logs.")
            services_are_initialized_successfully = False
            if doc_chat_service_instance and not os.path.exists(DEFAULT_DOCUMENTS_DIR):
                 print(f"  HINT: The documents directory '{DEFAULT_DOCUMENTS_DIR}' does not exist.")
            elif doc_chat_service_instance and os.path.exists(DEFAULT_DOCUMENTS_DIR) and not os.listdir(DEFAULT_DOCUMENTS_DIR) and (doc_chat_service_instance.faiss_index is None or getattr(doc_chat_service_instance.faiss_index, 'ntotal', 0) == 0):
                 print(f"  HINT: The documents directory '{DEFAULT_DOCUMENTS_DIR}' appears empty and no/empty index was loaded. Add documents and use 'Re-Index'.")

    except Exception as e:
        print(f"FLASK_APP_INIT: CRITICAL ERROR during DocumentChatService initialization: {e}")
        traceback.print_exc()
        services_are_initialized_successfully = False
        doc_chat_service_instance = DocumentChatService()

# Initialize services when the Flask app starts
initialize_core_services(rebuild=False)

# Main route, display the UI and document list
@app.route('/')
def index():
    global doc_chat_service_instance, services_are_initialized_successfully
    docs_in_index_list = []
    if doc_chat_service_instance and hasattr(doc_chat_service_instance, 'metadata') and doc_chat_service_instance.metadata:
        try:
            unique_sources = set()
            for _chunk_id, meta_info in doc_chat_service_instance.metadata.get("metadata_store", {}).items():
                source_name = meta_info.get("source_doc")
                if source_name:
                    unique_sources.add(source_name)
            docs_in_index_list = [{"name": src} for src in sorted(list(unique_sources))]
        except Exception as e:
            print(f"Error fetching document list for UI: {e}")
            traceback.print_exc()

    return render_template('index.html',
                           services_available=services_are_initialized_successfully,
                           documents=docs_in_index_list)

# Handles file uploads from the user
@app.route('/upload', methods=['POST'])
def upload_files():
    global doc_chat_service_instance
    #File types for upload
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not doc_chat_service_instance or not hasattr(doc_chat_service_instance, 'documents_dir'):
        flash("Core services (including document directory) not configured. Upload unavailable.", "error")
        return redirect(url_for('index'))

    if 'files[]' not in request.files:
        flash('No file part in request.', 'warning')
        return redirect(url_for('index'))

    files = request.files.getlist('files[]')
    if not files or (len(files) == 1 and files[0].filename == ''):
        flash('No files selected for upload.', 'info')
        return redirect(url_for('index'))

    upload_folder = doc_chat_service_instance.documents_dir
    try:
        os.makedirs(upload_folder, exist_ok=True)
    except OSError as e:
        flash(f"Could not create upload directory '{upload_folder}': {e}", "error")
        print(f"ERROR (upload_files): Could not create upload directory '{upload_folder}': {e}")
        return redirect(url_for('index'))
        
    uploaded_count = 0
    had_upload_attempts = False

    # Process each uploaded file
    for file in files:
        if file and file.filename:
            had_upload_attempts = True
            if allowed_file(file.filename): 
                filename = secure_filename(file.filename) 
                save_path = os.path.join(upload_folder, filename)
                try:
                    file.save(save_path)
                    uploaded_count += 1
                    print(f"Uploaded file: {save_path}")
                except Exception as e_save:
                    flash(f"Error saving file {filename}: {e_save}", "error")
                    print(f"Error saving file {filename}: {e_save}")
            else:
                flash(f"File type '{file.filename.rsplit('.', 1)[1] if '.' in file.filename else 'unknown'}' not allowed for {file.filename}.", "warning")
    
    if uploaded_count > 0:
        flash(f'{uploaded_count} file(s) uploaded to "{os.path.basename(upload_folder)}". Please use "Re-Index" for them to be included in search.', 'success')
    elif had_upload_attempts:
        flash('No files were successfully uploaded due to type restrictions or save errors.', 'info')
    
    return redirect(url_for('index'))

# Indexing of all documents
@app.route('/reindex', methods=['POST'])
def reindex_documents():
    global doc_chat_service_instance, services_are_initialized_successfully 
    flash("Re-indexing process started... This may take a moment.", "info")
    print("FLASK_APP: User triggered Re-indexing documents...")
    
    initialize_core_services(rebuild=True)
    
    if services_are_initialized_successfully and doc_chat_service_instance and hasattr(doc_chat_service_instance.faiss_index, 'ntotal'):
        flash(f"Re-indexing completed! Index now contains {doc_chat_service_instance.faiss_index.ntotal} chunks.", "success")
    else:
        flash("Re-indexing process finished, but there might be issues. Check server logs.", "warning")
        
    return redirect(url_for('index'))

# Process user queries
@app.route('/query', methods=['POST'])
def query_documents():
    global doc_chat_service_instance, services_are_initialized_successfully
    query = request.form.get('query', '').strip()
    llm_response_text = None
    doc_key_mapping = {}
    query_error_message = None

    if not query:
        flash("Please enter a query.", "warning")
        return redirect(url_for('index'))

    if not services_are_initialized_successfully or not doc_chat_service_instance:
        query_error_message = "Core chat services are not available. Cannot process query."
        flash(query_error_message, "error")
    else:
        try:
            # Get response from chat service
            print(f"FLASK_APP: Processing query: '{query}'")
            llm_response_text, doc_key_mapping = doc_chat_service_instance.process_query(query)
            if llm_response_text is None and doc_key_mapping is None:
                query_error_message = "Query processed but no response or references were generated."
            elif not llm_response_text and doc_key_mapping is not None: 
                query_error_message = "Could not get a synthesized response, but some documents were found."

        except Exception as e:
            flash(f"Error processing your query: {e}", "error")
            print(f"FLASK_APP: Error processing query '{query}': {e}")
            traceback.print_exc()
            query_error_message = f"An internal error occurred: {e}"

    docs_in_index_list = []
    if doc_chat_service_instance and hasattr(doc_chat_service_instance, 'metadata') and doc_chat_service_instance.metadata:
        try:
            unique_sources = set()
            for _chunk_id, meta_info in doc_chat_service_instance.metadata.get("metadata_store", {}).items():
                source_name = meta_info.get("source_doc")
                if source_name: unique_sources.add(source_name)
            docs_in_index_list = [{"name": src} for src in sorted(list(unique_sources))]
        except Exception as e_doclist:
            print(f"Error refreshing document list for UI: {e_doclist}")
            traceback.print_exc()

    return render_template('index.html',
                           services_available=services_are_initialized_successfully,
                           documents=docs_in_index_list,
                           last_query=query,
                           llm_response=llm_response_text,
                           doc_key=doc_key_mapping,
                           query_error=query_error_message
                           )

# Starts Flask development server
if __name__ == '__main__':
    print(f"Flask app CWD: {os.getcwd()}")
    
    print(f"Flask app template folder (resolved): {os.path.abspath(app.template_folder)}")
    if SERVICE_IMPORTS_OK:
      print(f"Attempting to use documents from (resolved): {os.path.abspath(DEFAULT_DOCUMENTS_DIR)}")
    else:
      print(f"Attempting to use DUMMY documents path (resolved): {os.path.abspath(DEFAULT_DOCUMENTS_DIR)}")
    app.run(debug=True, host='0.0.0.0', port=5000)