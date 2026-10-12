from dotenv import load_dotenv
import tempfile
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

def rag_basics():

    sample_data = ["Python is a programming language", "JavaScript is used for web development", 
    "Machine learning enables AI applications", "Deep learning uses neural networks", "Cats are popular pets",
    "LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.",
    "LangChain supports various LLM providers including OpenAI, Anthropic, and local models.",
    ]

    doc = [Document(page_content=t, metadata={"source": "sample_docs", "topic": "sample"}) for t in sample_data]

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(doc)

    print(f"Number of chunks created: {len(chunks)}")

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    model = init_chat_model(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    parser = StrOutputParser()

    question = "What is LangChain?"

    chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser

    result = chain.invoke(question)

    print(f"Question: {question}")
    print(f"Answer: {result}")

def rag_with_sources():

    sample_data = ["Python is a programming language", "JavaScript is used for web development", 
    "Machine learning enables AI applications", "Deep learning uses neural networks", "Cats are popular pets",
    "LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.",
    "LangChain supports various LLM providers including OpenAI, Anthropic, and local models.",
    ]

    doc = [Document(page_content=t, metadata={"source": "sample_docs", "topic": "sample"}) for t in sample_data]

    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_documents(doc)

    print(f"Number of chunks created: {len(chunks)}")

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Format retrieved docs with sources
    def format_docs_with_sources(docs):
        return "\n\n".join([f"{doc.page_content} (Source: {doc.metadata['source']})" for doc in docs])

    model = init_chat_model(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    parser = StrOutputParser()

    question = "What is LangChain?"

    chain = {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()} | prompt | model | parser

    result = chain.invoke(question)

    print(f"Question: {question}")
    print(f"Answer: {result}")    


if __name__ == "__main__":
    rag_basics()
    # rag_with_sources()