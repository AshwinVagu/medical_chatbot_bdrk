import json
import os
import sys
import boto3
import streamlit as st
import cv2
import pytesseract
from fpdf import FPDF

## We will be suing Titan Embeddings Model To generate Embedding

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="embeddings-model-id",client=bedrock) # Replace with your specific embeddings model ID

## Data ingestion
def data_ingestion():
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # - in our testing Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,
                                                 chunk_overlap=1000)
    
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_titan_llm():
    ##create the Titan Model
    llm=Bedrock(model_id="model-id",client=bedrock, # Replace with your specific model ID
                model_kwargs={'maxTokenCount':300, 'stopSequences':[], 'temperature':1, 'topP':1})
    return llm

# This is where the handwriting is recognized.
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text

# This returns the greyscale image
def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# This removes the noise in the image
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# This sets the threshold to turn an image black or white.
def set_thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# Function to create a PDF from text
def text_to_pdf(text, output_folder, filename="output.pdf"):
    # Create a PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font for the PDF (Arial, bold, size 12)
    pdf.set_font("Arial", size=12)
    
    # Insert the text into the PDF
    pdf.multi_cell(0, 10, text)
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the PDF in the specified folder
    output_path = os.path.join(output_folder, filename)
    pdf.output(output_path)
    
    return output_path



prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():

    st.set_page_config("Personal medical chatbot")
    
    st.header("Medico - Your hospital chatbot")

    user_question = st.text_input("Ask a Question about the hospital:")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

        if st.button("Read image file"):
            with st.spinner("Processing..."):
                img = cv2.imread('img.png')
                img = get_greyscale(img)
                img = remove_noise(img)
                img = set_thresholding(img)

                text = ocr_core(img)
                # Convert the text to a PDF and save it in the specified folder
                pdf_path = text_to_pdf(text, 'data', 'image.pdf')

                print(f"PDF file has been created at: {pdf_path}")
                st.success("Done")        

    if st.button("Output"):
        with st.spinner("Processing..."):
            #We use allow_dangerous_deserialization to ensure every file can go through for now.
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm=get_titan_llm()
            
            
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()
