import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from pypdf import PdfReader

st.set_page_config(page_title="DocuBot", page_icon="🤖")
st.title("🤖 DocuBot — AI Document Q&A")
st.write("Upload a PDF and ask questions about it!")

groq_api_key = st.text_input("Enter your Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and groq_api_key:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    with st.spinner("Processing your PDF..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("PDF processed! Ask your question below.")
    question = st.text_input("Ask a question about your PDF:")

    if question:
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        with st.spinner("Thinking..."):
            response = llm.invoke(
                f"Answer this question based only on the context below.\n\n"
                f"Context: {context}\n\nQuestion: {question}"
            )
            st.write("### Answer:")
            st.write(response.content)