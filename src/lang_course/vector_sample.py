from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def create_retriever(texts: list[str]):
    # 1. Convert text strings into Documents
    documents = [
        Document(
            page_content=text,
            metadata={"source": f"document_{i + 1}"}
        )
        for i, text in enumerate(texts)
    ]

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(documents)

    # 3. Create the embeddings model
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 4. Store the chunks in Chroma
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name="exercise_collection"
    )

    # 5. Return a configured retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever

sample_docs = [
    "Python is a programming language.",
    "JavaScript is used for web development.",
    "Machine learning enables AI applications.",
    "Deep learning uses neural networks.",
    "LangChain helps developers build applications powered by language models."
]

if __name__ == "__main__":
    retriever = create_retriever(sample_docs)
    results = retriever.invoke(
        "What helps developers build LLM applications?"
    )


    for doc in results:
        print("Source:", doc.metadata["source"])
        print("Content:", doc.page_content)
        print("-" * 40)