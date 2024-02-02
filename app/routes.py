from flask import Flask, Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import os
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer

import openai
import pprint 

from dotenv import load_dotenv

import pickle


import faiss
import numpy as np

load_dotenv(dotenv_path=".env")


pp = pprint.PrettyPrinter(indent=4)


main = Blueprint("main", __name__, url_prefix="/api")



OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


@main.route('/upload-pdf', methods=['POST'])
def upload_file():
    req_form = request.form.to_dict()
    user_info = eval(req_form["user"])  # Note: Using eval is dangerous and not recommended
    print(user_info)
    print("user info:", type(user_info))
    if not user_info:
        return "No User", 400

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Secure filename
        filename = secure_filename(file.filename)
        # Create directories if they do not exist
        user_dir = os.path.join('files', user_info['username'])
        document_dir = os.path.join(user_dir, os.path.splitext(filename)[0])
        os.makedirs(document_dir, exist_ok=True)
        # Complete filepath
        filepath = os.path.join(document_dir, filename)
        print("filepath:", filepath)
        # Save file
        file.save(filepath)

        # Process PDF to extract text
        extracted_text = extract_text_from_pdf(filepath)

        # Further processing or storing the extracted text
        # ...

        return 'File uploaded and processed', 200


def extract_text_from_pdf(filepath):
    pdf_reader = PdfReader(filepath)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
    )
    
    # You may want to define your batch size based on your system's memory constraints
    batch_size = 2  # Example batch size, adjust based on your requirements
    current_batch = []
    items = [] 
    
    for page_number, page in enumerate(pdf_reader.pages):
        for position, chunk in enumerate(text_splitter.split_text(page.extract_text())):
            text_object = {
                "text": chunk,
                "page_number": page_number,
                "position": position,
            }
            # items.append(text_object)

            # When the batch reaches the batch size, process it
            vector = vectorize(chunk)
            text_object["vector"] = vector
            items.append(text_object)

    
    pickle_file = filepath
    picklefile = pickle_file.replace(".pdf", ".pkl")
    print("trying to store pickle @:", picklefile)
    with open(picklefile, 'wb') as f: 
        pickle.dump(items, f)



def vectorize(chunk):
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    # return openai.embeddings.create(
    #     input=chunk,
    #     model="text-embedding-3-small"
    # ).data[0].embedding
    return model.encode(chunk).tolist()
  
   



@main.route("/get_embeddings", methods=["GET"])
def get_stored_embeddings():
    # Get the file path from the query parameters
    filepath = request.args.get('filepath')
    
    # Check if the file exists
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    print("path to pdf found")
    # The pickle file is assumed to be in the same directory as the PDF, with a .pkl extension
    pickle_file = filepath.replace('.pdf', '.pkl')
    print("path to pkl = " , pickle_file)
    # Check if the pickle file exists
    if not os.path.exists(pickle_file):
        return jsonify({"error": "Embeddings not found"}), 404

    # Load the embeddings from the pickle file
    with open(pickle_file, 'rb') as f:
        embeddings = pickle.load(f)
    

    
    # Return the embeddings as JSON
    return jsonify(embeddings)




def load_faiss_index(pickle_file):
    with open(pickle_file, 'rb') as f:
        items = pickle.load(f)
    # Assuming all vectors are of the same size
    d = len(items[0]["vector"]) if items else 0
    index = faiss.IndexFlatL2(d)  # Create a flat (brute force) index
    vectors = np.array([item["vector"] for item in items]).astype('float32')
    index.add(vectors)  # Add vectors to the index
    return index, items

# API endpoint for search
@main.route('/search', methods=['POST'])
def search_embeddings():
    data = request.json
    query_text = data.get('query')
    filepath = data.get("filepath")
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    # Convert the query text to a vector using the same method as for the documents
    query_vector = vectorize(query_text)

    # Load the index and items from the pickle file
    # Make sure to pass the correct path to your pickle file
    pickle_file = filepath
    faiss_index, items = load_faiss_index(pickle_file)

    # Convert the query vector to the right type and run the search
    query_vector = np.array([query_vector]).astype('float32')
    _, I = faiss_index.search(query_vector, 1)  # Search for the top 1 nearest vector

    # Find the closest text object and return it
    closest_item_index = I[0][0]
    closest_item = items[closest_item_index]

    return jsonify(closest_item)


from transformers import pipeline

# Load your question-answering model
qa_model = pipeline("question-answering")

@main.route("/search-hugging-face", methods=["POST"])
def search_embeddings_hf():
    data = request.json
    query_text = data.get('query')
    filepath = data.get("filepath")

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    # Convert the query text to a vector using the same method as for the documents
    query_vector = vectorize(query_text)

    # Load the index and items from the pickle file
    pickle_file = filepath
    faiss_index, items = load_faiss_index(pickle_file)

    # Convert the query vector to the right type and run the search
    query_vector = np.array([query_vector]).astype('float32')
    _, I = faiss_index.search(query_vector, 5)  # Adjust the number based on your needs

    # Retrieve and concatenate texts from the closest items for context
    context = " ".join([items[i]['text'] for i in I[0]])

    # Use the QA model to find the answer
    answer = qa_model(question=query_text, context=context)

    return jsonify({"answer": answer['answer'], "confidence": answer['score']})

