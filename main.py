#This section imports various libraries needed for different functionalities like building the web app, handling data, and working with language models.
import os # Provides functions to interact with the operating system
import streamlit as st  # Streamlit for building the web app
import pickle # For serializing and deserializing data
import base64  # For encoding and decoding base64
from langchain import OpenAI # For using the OpenAI language model
from langchain.chains import RetrievalQAWithSourcesChain # For question-answering with sources
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into chunks
from langchain.document_loaders import UnstructuredURLLoader # For loading documents from URLs
from langchain.embeddings import OpenAIEmbeddings # For creating embeddings using OpenAI
from langchain.vectorstores import FAISS # For creating and managing a FAISS index
from dotenv import load_dotenv # For loading environment variables from a .env file
# Load environment variables from .env (especially OpenAI API key)
load_dotenv()

#  Function to convert image to base64 format for embedding in HTML/CSS
def get_background_image_url(image_path):
    with open(image_path, "rb") as img_file:
        return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"

# Set background image in CSS
background_image_url = get_background_image_url("background.jpg")

# Adding Custom CSS (styling) for the Streamlit app
st.markdown(f"""
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-image: url('{get_background_image_url("background.jpg")}');
            background-size: cover;
            background-position: center;
            color: #fff;
        }}
        .title {{
            color: rgba(255, 255, 255, 0.8);
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
            animation: fadeInScale 2s forwards;
        }}
        .subtitle {{
            color: rgba(255, 255, 255, 0.6);
            font-size: 20px;
            text-align: center;
            animation: fadeIn 2s forwards;
            opacity: 0;
            margin-bottom: 30px;
        }}
        @keyframes fadeInScale {{
            from {{ opacity: 0; transform: scale(0.5); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .sidebar {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
        }}
        .sidebar input[type="text"] {{
            margin-bottom: 10px;
            width: 100%;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
            font-size: 18px;
            height: 150px;
            box-sizing: border-box;
        }}
        .sidebar button {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
        }}
        .sidebar button:hover {{
            background-color: #0056b3;
        }}
        .content {{
            margin: 20px;
        }}
        .content .header {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .content .subheader {{
            font-size: 20px;
            color: #333;
            margin-top: 10px;
        }}
        .content p {{
            line-height: 1.6;
            color: #333;
        }}
        .answer {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .alert {{
            color: red;
            font-size: 16px;
            text-align: center;
            margin: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

# Title and Subtitle with Animations
st.markdown('<div class="title">Welcome to NewsBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A place to understand your news articles better</div>', unsafe_allow_html=True)

# Sidebar for entering the URLs
st.sidebar.title("Enter the News Article URLs")

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    if not any(urls):
        st.sidebar.markdown('<div class="alert">Please Enter the News Article\'s URL</div>', unsafe_allow_html=True)
    else:
        # Load and process data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading Started") # Display a message indicating that data loading has started
        data = loader.load() # Load the data from the URLs
        # Check if data was loaded successfully
        if not data:
            st.sidebar.markdown('<div class="alert">Failed to load data. Please check the URLs.</div>', unsafe_allow_html=True)
            st.stop()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','], # Define separators for splitting the text (paragraphs, lines, sentences)
            chunk_size=1000 # Set the maximum chunk size to 1000 characters
        )
        main_placeholder.text("Text Splitter Initiated")# Display a message indicating that text splitting has started
        docs = text_splitter.split_documents(data)  # Split the data into chunks

        if not docs:
            st.sidebar.markdown('<div class="alert">Failed to split documents. Data might be empty.</div>', unsafe_allow_html=True) # Show an error if splitting failed
            st.stop()

        #  Create embeddings (numerical representations) of the text and store them in a FAISS index for fast search
        embeddings = OpenAIEmbeddings() # Initialize the embedding model
        try:
            vectorstore_openai = FAISS.from_documents(docs, embeddings)  # Create a FAISS index from the document chunks
            main_placeholder.text("Embedding Vector Building Started") # Display a message indicating that embedding creation has started
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f) # Save the FAISS index to a file
            main_placeholder.text("FAISS index saved successfully.") # Display a success message
        except Exception as e:
            st.sidebar.markdown(f'<div class="alert">Error creating FAISS index: {e}</div>', unsafe_allow_html=True) # Show an error message if something went wrong
            st.stop()# Stop further execution
# Input field for the user to enter a question
query = main_placeholder.text_input("Enter your question")  # Display a text input field for the user to ask a question
if query:
    if os.path.exists(file_path):  # Check if the FAISS index file exists
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f) # Load the saved FAISS index from the file
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())  # Create a chain to handle Q&A with sources
            result = chain({"question": query}, return_only_outputs=True) # Pass the user's question to the chain and get the answer
            st.header("Answer")  # Display the header for the answer section
            st.write(result["answer"])   # Show the answer

            # Display sources, if available
            sources = result.get("sources", "") # Get the sources (if any) from the result
            if sources:
                st.subheader("Sources:") # Display the header for the sources section
                for source in sources.split("\n"):  # Split the sources by line
                    st.write(source) # Display each source
