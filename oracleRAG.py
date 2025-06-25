"""
Oracle AI Vector Search RAG (Retrieval-Augmented Generation) System
===================================================================

This script implements a complete RAG pipeline using Oracle Database 23ai's native vector search
capabilities with LangChain integration. It processes PDF documents, generates embeddings,
stores them in Oracle's vector database, and provides AI-powered question answering.

Key Components:
- PDF Document Processing: Extracts and chunks text from Oracle AI Vector Search documentation
- Vector Embeddings: Generates embeddings using HuggingFace or Nomic embedding models
- Oracle Vector Store: Stores embeddings in Oracle Database 23ai using VECTOR data type
- RAG Pipeline: Combines vector similarity search with LLM for contextual answers
- Multiple Database Support: Configured for both local Oracle DB and Autonomous Database

Features:
- Configurable chunk sizes and overlap for optimal retrieval performance
- Support for multiple embedding models (HuggingFace, Nomic)
- L2 normalization option for improved vector search accuracy
- Cosine similarity search using Oracle's native vector operations
- Integration with Ollama LLM for local AI inference
- Automatic table creation and data management

Database Configuration:
- Local Oracle Database (FREEPDB1)
- Oracle Autonomous Database with wallet authentication
- Vector table with VECTOR({VECTOR_DIMENSION}) column type

Dependencies:
- oracledb: Oracle Database connectivity
- langchain: Document processing and RAG pipeline
- numpy: Vector operations and normalization
- huggingface-embeddings: Text embedding generation

Usage:
1. Configure database connection parameters
2. Set embedding model and vector dimensions
3. Run script to process documents and create vector store
4. Query the system using natural language questions

Input: Oracle AI Vector Search PDF documentation
Output: AI-powered answers based on document content with vector similarity search

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""
import array
import oracledb
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_nomic import NomicEmbeddings


PDF_URL = "https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/ai-vector-search-users-guide.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20

VECTOR_TABLE = "hf_emb"
### EMBEDDING MODELS
### HUGGINGFACE
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
### NOMIC
#EMBEDDING_MODEL = "nomic-embed-text-v1.5"
#VECTOR_DIMENSION = 768

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def process_documents():

    # Connect to ADB
    # connection = oracledb.connect(
    #     user="admin",
    #     password="Password",
    #     dsn="mydb_high",
    #     config_dir="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_location="/Users/dev/Wallets/Wallet_mydb",
    #     wallet_password="Password"
    # )

    # Connect to local Container
    connection = oracledb.connect(
        user="testuser",
        password="Password",
        dsn="localhost:1521/FREEPDB1"
    )

    # Load and split PDF
    loader = PDFMinerLoader(PDF_URL)
    documents = loader.load()
    print(f"Loaded {len(documents)} document sections")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")

    # Embedding models
    # HUGGINGFACE
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    # NOMIC
    #os.environ["NOMIC_API_KEY"] = "nk-qF7wOnp7aSFv8eNnCT9cNwZpE23RKpoc-MvIWcNs01w"
    #embeddings_model = NomicEmbeddings(model=EMBEDDING_MODEL)


    texts = [chunk.page_content if hasattr(chunk, "page_content") else str(chunk) for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
    # This line ensures that every embedding is a NumPy array of 32-bit floats,
    # which is the required format for Oracle's vector search functionality in 23ai.
    embeddings = [np.asarray(emb, dtype=np.float32) for emb in embeddings]

    # Apply L2 normalization to each embedding
    #embeddings = [l2_normalize(emb) for emb in embeddings]

    print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")

    # Create table if not exists
    with connection.cursor() as cursor:
        cursor.execute(f"""
           BEGIN
                EXECUTE IMMEDIATE '
                    CREATE TABLE {VECTOR_TABLE} (
                        id NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY,
                        embedding VECTOR({VECTOR_DIMENSION}),
                        text CLOB,
                        metadata CLOB,
                        CONSTRAINT {VECTOR_TABLE}_pk PRIMARY KEY (id)
                    )
                ';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
            END;
        """)

    # Insert embeddings and text
    oracle_vectors = [array.array('f', emb) for emb in embeddings]
    data = list(zip(oracle_vectors, texts))
    with connection.cursor() as cursor:
        cursor.executemany(
            f"INSERT INTO {VECTOR_TABLE} (embedding, text) VALUES (:1, :2)",
            data
        )
        connection.commit()

    # Set up retriever and LLM
    vector_store = OracleVS(
        client=connection,
        embedding_function=embeddings_model,
        table_name=VECTOR_TABLE,
        distance_strategy=DistanceStrategy.COSINE
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    llm = ChatOllama(model="llama3.1:latest")

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n{context}\nQuestion: {question}\n"
    )

    chain = (
        {"context": (lambda x: x["question"]) | retriever,
         "question": (lambda x: x["question"])}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "What are Vector Indexes?"
    answer = chain.invoke({"question": question})
    print("Answer:", answer)

    connection.close()

if __name__ == "__main__":
    process_documents()

