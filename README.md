# Medico: A Medical Chatbot for Hospitals

## Overview

Welcome to the Medico repository! This project features two different chatbots designed to assist with hospital operations. The first is a basic chatbot, and the second is an advanced Retrieval-Augmented Generation (RAG) chatbot, integrated with hospital-specific data extracted from PDF files. The RAG chatbot is powered by Amazon's Titan LLM (`amazon.titan-text-lite-v1`) and is capable of answering detailed questions related to hospital operations.

## Features

- **Basic Chatbot**: A simple chatbot implementation to handle general conversations.
- **RAG Chatbot**: A sophisticated chatbot that leverages a Retrieval-Augmented Generation framework to answer questions based on hospital-specific documents.
- **Text Extraction**: The RAG chatbot can read and extract text from scanned handwritten images, convert them to PDFs, and utilize this information in responses.
- **AWS Bedrock**: Utilizes Amazon's Bedrock service to interact with the `amazon.titan-text-lite-v1` model for generating responses.
- **FAISS Vector Database**: Efficiently searches through the hospital documents by storing them as vectors in a FAISS index.
- **Streamlit Frontend**: Provides an interactive frontend for users to engage with the chatbot.

## Requirements

Make sure you have pip, python and Amazon CLI setup in your environment.

Before initializing either of the chatbots, ensure you have installed all the necessary libraries by running the following command:

pip install -r requirements.txt



Libraries in requirements.txt:

boto3: AWS SDK for Python to interact with Amazon Bedrock.
streamlit: Used for creating the web-based user interface.
opencv-python: For image processing tasks such as reading and manipulating images.
pytesseract: Optical Character Recognition (OCR) tool for extracting text from images.
fpdf: Library for generating PDF files from extracted text.
langchain: For building the RAG framework and handling LLM interactions.
faiss-cpu: For efficient similarity search and clustering using FAISS.
numpy: For handling numerical operations.


How to Run:

1. Basic Chatbot
To initialize the basic chatbot, run:

python basic_chatbot.py

2. RAG Chatbot for the Hospital
To initialize the advanced RAG chatbot designed for the hospital:

streamlit run medical_rag.py

This will start the Streamlit application, where you can interact with the chatbot via a web interface.


Project Structure:

basic_chatbot.py: Contains the implementation of the basic chatbot.
medical_rag.py: Contains the implementation of the RAG chatbot that uses Amazon's Titan model and the FAISS vector store.
requirements.txt: Lists all the necessary Python libraries to be installed.
data/: Directory where hospital PDF files and generated PDFs from scanned images are stored.
faiss_index/: Directory where the FAISS index files are saved.



Key Functionalities:

- RAG Chatbot 
- Data Ingestion: The chatbot ingests PDF files located in the data/ directory, splits the documents into chunks, and stores them in a FAISS vector database.
- Question Answering: The RAG framework retrieves relevant document chunks using FAISS and uses the Titan LLM to generate detailed responses.
- Text Extraction from Images: The chatbot can extract text from handwritten images using pytesseract, convert the text to a PDF, and add it to the document database for future queries.



Usage Instructions

1. Upload PDFs
Place your hospital's operational PDFs in the data/ directory. These files will be ingested and used by the RAG chatbot to provide accurate and context-specific answers.

2. Update Vector Store
Use the sidebar in the Streamlit application to update the vector store with newly ingested documents.

3. Handwritten Text Extraction
Upload handwritten scanned images, and the chatbot will extract the text, convert it to a PDF, and add it to the database for future queries.



Contributing:
If you'd like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions of all kinds, including bug fixes, feature additions, and documentation improvements.


### Note:
Ensure that all the libraries mentioned in the provided code snippets are included in your `requirements.txt`. If any are missing, add them to avoid runtime errors.

This `README.md` provides a comprehensive overview of the project, including installation, usage instructions, and key functionalities. It should serve as a useful guide for anyone looking to use or contribute to your project.