# ----- required imports -----

import json
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----- helper functions -----


def load_corpus(filepath):
    """
    reads the local corpus json file and
    returns the data
    """
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return None


def chunk_corpus():
    """
    processes corpus using chunking
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = [entry["text"] for entry in data]
    return text_splitter.split_texts(texts)


def create_vector_store(texts):
    """
    creates a vector store for
    effective context retrieval
    """
    embeddings = OpenAIEmbeddings()  # Use appropriate embeddings for your model
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def load_model(model_name):
    """
    attempts to load the specified
    ollama client model
    """
    try:
        model = Ollama(model_name=model_name)
        return (True, model)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return (False, None)


def query_model(model, vector_store, question):
    """
    generates a query that is then
    used to prompt the model
    """
    relevant_texts = vector_store.similarity_search(question, k=3)  # Adjust k as needed
    context = "\n".join([text.page_content for text in relevant_texts])
    prompt = f"{context}\n\n{question}"
    response = model(prompt)
    return response


# ----- sample execution -----

if __name__ == "__main__":
    TARGET_FILEPATH = "./../corpus/clean/tort/corpus.json"
    MODEL_NAME = "your_model_name"
    data = load_corpus(TARGET_FILEPATH)
    if data is not None:
        success, model = load_model(MODEL_NAME)
        if success and model is not None:
            texts = chunk_corpus(data)
            vector_store = create_vector_store(texts)
            question = (
                "What are the implications of Ollie's programming on human safety?"
            )
            response = query_model(model, vector_store, question)
            print(f"Model Response: {response}")
        else:
            print("Error: Failed to load the model.")
    else:
        print("Failed to load the corpus.")
