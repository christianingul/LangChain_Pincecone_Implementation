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
    pinecone.init(api_key='82e61102-f1d3-4a95-9ce3-8dd53f7dfa3b', environment='asia-southeast1-gcp-free')

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

    embeddings = OpenAIEmbeddings(openai_api_key="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")
    docsearch = Pinecone.from_texts(chunks, embeddings, index_name="pdf-embeddings-index")

    qa = VectorDBQA.from_chain_type(llm=OpenAI(model_name='gpt-3.5-turbo', openai_api_key="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"),
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
    openai = OpenAI(openai_api_key="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", temperature=0, model_name='gpt-3.5-turbo')
    csv_agent = create_csv_agent(openai, path=user_csv, verbose=True)
    return csv_agent

#Here the agent and the user question are run
def run_agent(agent, user_question):
    response = agent.run(user_question)
    return response

if __name__ == '__main__':
    main()
