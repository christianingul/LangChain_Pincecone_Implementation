import streamlit as st
from PyPDF2 import PdfReader
from langchain.agents import create_csv_agent
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone

def main():
    st.set_page_config(page_title="Ask Your Own data :D")
    st.header("Let an LLM Interact With Your Own data!")

    password_secret = st.secrets.get("PASSWORD")
    openai_secret = st.secrets.get("openai")
    pinecone_secret = st.secrets.get("pinecone")
    pinecone_env = st.secrets.get("environment")

    if password_secret is None or openai_secret is None or pinecone_secret is None or pinecone_env is None:
        st.error("Required secrets are missing. Please check your secrets configuration.")
        st.stop()

    password = st.text_input("Enter password:", type="password")
    if password != password_secret.get("password"):
        st.error("Reach out to cingul@usc.edu for a password")
        st.stop()

    file_type = st.radio("Select file type:", ("CSV", "PDF"))

    if file_type == "CSV":
        csv_file_handling()
    elif file_type == "PDF":
        pdf_file_handling()

def pdf_file_handling():
    st.markdown('### PDF Question & Answering 💬')
    st.markdown('''
        Upload your PDF file using the file uploader below. Make sure the PDF file contains the relevant data you want to ask questions about.
        
        What goes on behind the scenes?
        
        1. First, the user-provided PDF file is split into text chunks in order to prepare it to be embedded (this is so we can stay below OpenAI's token limit).
        2. Followingly, the text chunks are passed into OpenAI's embedding model, where text chunks are converted to vectors.
        3. The vectors derived from the embedding process are uploaded into a vector database (Pinecone).
        4. The user input (question) is now passed into Pinecone's vector database as a query, which goes through the same cycle as the PDF.
        5. We then perform a similarity search using the query vector to find relevant vectors. Finally, Langchain extracts the relevant vectors identified by the similarity search and converts them back into text chunks.
        6. Lastly, the relevant text chunks are passed along with the user question to the LLM. Now, the LLM has both the relevant information and the question.
                
        **Best Practices:**

        - Use PDF files that are text-based rather than scanned images. Text-based PDFs provide better results as the text can be extracted accurately.
        - Verify that the PDF is properly formatted and readable. In some cases, poorly formatted or corrupted PDFs may not yield accurate results.
        - NOTE, the LLM answer is limited to the quality of your prompt, so please spend time prompt engineering. Also, the LLM can only answer questions related to your upload.
        - IMPORTANT: You are querying a Vector DB, so questions need to be specific: If I upload a resume, "What are person xxx skills?"
        ''')
    user_pdf = st.file_uploader("Upload your PDF file", type="pdf")

    if user_pdf is not None:
        user_question = st.text_input("Ask a question about your data:")

        if user_question:
            response = process_pdf_and_run_agent(user_pdf, user_question)
            st.write(response)

def process_pdf_and_run_agent(user_pdf, user_question):
    openai_secret = st.secrets.get("openai")
    pinecone_secret = st.secrets.get("pinecone")
    pinecone_env = st.secrets.get("environment")

    openai_api_key = openai_secret.get("key")
    pinecone_api_key = pinecone_secret.get("key")
    pinecone_environment = pinecone_env.get("key")

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

    qa = VectorDBQA.from_chain_type(llm=ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=openai_api_key
),
                                    chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    response = qa({"query": user_question})

    #pinecone.deinit()

    return response


def csv_file_handling():
    st.markdown('### CSV Agent')
    st.markdown('''
        Upload your CSV file using the file uploader below. Ensure that the CSV file contains structured data that can be processed by the tool.
        
        What happens when you upload a CSV file? Behind the scenes, LangChain's panda's data frame agent is running. The agent can be thought of as a tool that can interact with its environment and make its own decisions. Underneath the hood of the agent is an LLM.
        
        CSV Agent Flow:
        
        1. The LangChain panda's data frame agent receives an input (question); to understand what to do, it passes the input to an LLM (its own reasoning engine).
        2. The reasoning engine (LLM) will think of a solution. In our case, it realizes it needs to perform an action, and that is to access the CSV file provided by the user.
        3. Further, the LLM will realize from the agent's tool description list (the LLM is provided the list with the user question), that it can access the CSV file through the pandas data frame tool.
        5. After the LLM executes some input (python code) through the pandas data frame tool (agent), the output is observed, and if it is satisfactory, the results are displayed to the user.
        6. However, this doesn't always happen; in that case, the process will restart, and the LLM could choose to try a different tool and strategy.
        7. This process continues until a stopping criterion (often a reasonable answer) is met and the results are displayed.

        **Best Practices:**

        - Organize your data in a tabular format, with each column representing a specific attribute or feature.
        - Make sure your CSV file doesn't contain unnecessary headers, footers, or extraneous information that could affect the results.
        - Check that your CSV file doesn't have any missing or inconsistent values, as they may impact the accuracy of the responses.
        - NOTE, the LLM answer is limited to the quality of your prompt, so please spend time prompt engineering. Also, the LLM can only answer questions related to your upload.
        
        Backend Script: https://github.com/christianingul/LangChain_Pincecone_Implementation/blob/main/app.py
        ''')

    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        user_question = st.text_input("Ask a question about your data:")

        if user_question:
            agent = create_agent(user_csv)
            response = run_agent(agent, user_question)
            st.write(response)

#This one is linked back to the .csv_py, where it takes the user_csv, and installs the pandas_agent.
def create_agent(user_csv):

    openai_secret = st.secrets.get("openai")
    openai_api_key = openai_secret.get("key")


    openai = ChatOpenAI(openai_api_key=openai_api_key
                    , temperature=0, model_name='gpt-3.5-turbo')
    csv_agent = create_csv_agent(openai, path=user_csv, verbose=True)
    return csv_agent

#Here the agent and the user question are run
def run_agent(agent, user_question):
    response = agent.run(user_question)
    return response

if __name__ == '__main__':
    main()
