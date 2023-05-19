import streamlit as st
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

def main():
    st.set_page_config(page_title="Ask your Data")
    st.header("Ask your Data 💬")

    password_secret = st.secrets.get("PASSWORD")
    openai_secret = st.secrets.get("openai")
    pinecone_secret = st.secrets.get("pinecone")
    pinecone_env = st.secrets.get("environment")

    if password_secret is None or openai_secret is None or pinecone_secret is None or pinecone_env is None:
        st.error("Required secrets are missing. Please check your secrets configuration.")
        st.stop()

    password = st.text_input("Enter password:", type="password")
    if password != password_secret.get("password"):
        st.error("You can reach out to cingul@usc.edu for the password")
        st.stop()

    file_type = st.radio("Select file type:", ("CSV", "PDF"))

    if file_type == "CSV":
        csv_file_handling()
    elif file_type == "PDF":
        pdf_file_handling()

def pdf_file_handling():
    st.markdown('### PDF File Handling')
    user_pdf = st.file_uploader("Upload your PDF file", type="pdf")

    if user_pdf is not None:
        user_question = st.text_input("Ask a question about your data:")

        if user_question:
            response = process_pdf_and_run_agent(user_pdf, user_question)
            st.write(response)

def process_pdf_and_run_agent(user_pdf, user_question):

    openai_api_key = st.secrets["openai"]
    pinecone_api_key = st.secrets["pinecone"]
    pinecone_environment = st.secrets["environment"]

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    pdf_reader = PdfReader(user_pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    #Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key
)
    docsearch = Pinecone.from_texts(chunks, embeddings, index_name="pdf-embeddings-index")

    qa = VectorDBQA.from_chain_type(llm=OpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key
),
                                    chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    response = qa({"query": user_question})

    #pinecone.deinit()

    return response


def csv_file_handling():
    st.markdown('### CSV File Handling')
    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your data:")

        if user_question:
            agent = create_agent(user_csv)
            response = run_agent(agent, user_question)
            st.write(response)

#This one is linked back to the .csv_py, where it takes the user_csv, and installs the pandas_agent.
def create_agent(user_csv):

    openai_api_key = st.secrets["openai"]

    openai = OpenAI(openai_api_key=openai_api_key
                    , temperature=0, model_name='gpt-3.5-turbo')
    csv_agent = create_csv_agent(openai, path=user_csv, verbose=True)
    return csv_agent

#Here the agent and the user question are run
def run_agent(agent, user_question):
    response = agent.run(user_question)
    return response

if __name__ == '__main__':
    main()
