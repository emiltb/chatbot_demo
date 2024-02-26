import gradio as gr
import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4-0125-preview",
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Indlæs vektor-database hvis den allerede findes, ellers danner vi den
if Path("faiss_index").exists():
    vector = FAISS.load_local("faiss_index", embeddings_model)
else:
    loader = DirectoryLoader("queens_speeches/speeches", glob="**/*.txt")
    docs = loader.load()

    for i, doc in enumerate(docs):
        doc.metadata["year"] = int(
            doc.metadata["source"].split("/")[-1].split(".")[0]
        )

    text_splitter = SemanticChunker(embeddings_model)
    documents = text_splitter.split_documents(docs)

    vector = FAISS.from_documents(documents, embeddings_model)
    vector.save_local("faiss_index")

# Definer prompt, som ChatGPT tager udgangspunkt i for at besvare spørgsmål
prompt = ChatPromptTemplate.from_template(
    """Besvar nedenstående spørgsmål, baseret på den givne kontekst og din generelle viden om den royale familie. Vær særligt opmærksom på relationer, datoer og titler:

<kontekst>
{context}
</kontekst>

Spørgsmål: {input}"""
)

# Bind model og prompt sammen og lav så en retrieval_chain, som kan bruges til at stille spørgsmål
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def dronning_chat(message, history):
    response = retrieval_chain.invoke({"input": message})
    return response["answer"]


gr.ChatInterface(
    dronning_chat, title="twoday Kapacity EGN OpenAI hackathon"
).launch()
