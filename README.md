# NewsBot
News Article Analyzer



This project is a Streamlit-based web app named NewsBot that processes and analyzes news articles using the OpenAI language model and FAISS for efficient text retrieval. Users can input URLs of news articles, which are then loaded and split into chunks. The text is embedded using OpenAIEmbeddings, and a FAISS index is created to enable quick question-answering with sources. The app also features custom CSS for a polished UI and supports environment variable management via dotenv. Dependencies include langchain, streamlit, and faiss-cpu. The app is designed for user-friendly interaction, making it easy to explore news content.

References:

https://python.langchain.com/v0.2/docs/introduction/

https://docs.streamlit.io/

https://faiss.ai/ | https://pypi.org/project/faiss-cpu/ 

https://developer.mozilla.org/en-US/docs/Web/CSS

