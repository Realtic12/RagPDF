import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import tempfile
from pypdf import PdfReader

# Set the title and caption of the Streamlit app
st.title("Chat with PDF 🌐")
st.caption("This app allows you to chat with a PDF using local llama3 and RAG")

# Get the PDF file from the user
pdf_file = st.file_uploader("Upload a PDF File", type="pdf")

if pdf_file:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(pdf_file.getvalue())
        temp_pdf_path = f.name

    # Load the PDF and extract text from each page
    reader = PdfReader(temp_pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split the extracted text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    print(text_splitter)
    splits = text_splitter.split_text(text)

    # Create embeddings and a vector store from the text chunks
    embeddings = OllamaEmbeddings(model="llama3.1")
    vectorstore = Chroma.from_texts(texts=splits, embedding=embeddings)

    # Define a function to query the Ollama Llama3 model
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    # Set up the retriever from the vector store
    retriever = vectorstore.as_retriever()

    # Combine the retrieved documents into a single string
    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG (Retrieval-Augmented Generation) chain function
    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)

    st.success("PDF loaded successfully!")

    # Provide an input box for the user to ask questions about the PDF
    prompt = st.text_input("Ask any question about the PDF")

    # Display the answer from the RAG chain function
    if prompt:
        result = rag_chain(prompt)
        st.write(result)