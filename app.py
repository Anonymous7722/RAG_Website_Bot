from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import streamlit as st


load_dotenv()

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

llm  = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    collection_name="Rag_Collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)



def AI_OUTPUT(vector_store,user_input):
    system_promt = (
    "You are an assistant for Question-answering tasks. Use the following website pieces of retrieved data to answer the question. If you don't know the answer, decline politely. and keep the answer concise. {context} any programming and dveelopment related query"
    )

    promt = ChatPromptTemplate.from_messages(
        [
            ("system", system_promt),
            ("human", "{input}")
        ]
    )

    llm_chain = promt | llm | StrOutputParser()

    llm_output = llm_chain.invoke({'context':vectorstore.similarity_search(user_input, k=5),'input':user_input})
    st.write(llm_output)






st.title('Website Summarizer ( Q&A )')

user_web_path = st.text_input('Enter your Website URL')
user_input_query = st.text_input('Enter your Query related to the Website')


if st.button('Send'):
    loader = WebBaseLoader(web_paths=[user_web_path])
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    splits = text_splitter.split_documents(docs)


    print(f"No. of Split of the given website: {len(splits)}")
    vectorstore =  Chroma.from_documents(documents=docs,embedding=embeddings)
    AI_OUTPUT(vector_store,user_input_query)