# ChatBot with PDF

This Streamlit app allows users to interact with a PDF document using the local Llama3 model and Retrieval-Augmented Generation (RAG). The app takes a PDF file as input, processes it, and enables users to ask questions about the content of the PDF, receiving answers generated by the Llama3 model (by default, user can change it) with relevant context from the document.

## Features

- Upload a PDF file to the app.
- Extract text from the PDF and split it into chunks for processing.
- Create embeddings from the text chunks using the Llama3 model.
- Use a vector store to retrieve relevant context for user queries.
- Answer user questions about the PDF content using RAG with the Llama3 model.

## Installation

It should be suitable to use a virtual environment, like Conda to avoid future issues with packages.

```bash
git clone https://github.com/Realtic12/RagPDF.git
cd RagPDF
pip install requirements.txt
```

## Usage

1. Run the app
```bash
streamlit run Local_rag.py
```

2. Upload a PDF

3. Ask questions about the uploaded document
