import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import faiss
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()
# os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')
os.environ["Groq_Api_Key"]= GROQ_API_KEY

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text


def get_text_chunks(text):
    tex_splitters = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks= tex_splitters.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vector_store = faiss.FAISS.from_texts(text_chunks, embedding)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context use your own knowledge with the provided context to answer the query\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGroq(
    model="gemma-7b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
    prompt=PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt= prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    new_db = faiss.FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                raw_text = get_pdf_text(pdf_docs) # get the pdf text
                text_chunks = get_text_chunks(raw_text) # get the text chunks
                get_vector_store(text_chunks) # create vector store
                st.success("Done")
        
        st.write("---")
        
        st.write("AI App created by @ Ashir Mumtaz")  # add this line to display the image


    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/ashirmumtaz" target="_blank">Ashir Mumtaz</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()