from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil


load_dotenv()

SAMPLE_DOCS = [
    Document(page_content="Python is a programming language",
    metadata={"source": "python_docs", "topic": "programming"}),
    Document(page_content="JavaScript is used for web development",
    metadata={"source": "javascript_docs", "topic": "database"}),
    Document(page_content="Machine learning enables AI applications",
    metadata={"source": "ml_docs", "topic": "machine_learning"}),
    Document(page_content="Deep learning uses neural networks",
    metadata={"source": "dl_docs", "topic": "deep_learning"}),
    Document(page_content="Cats are popular database pets",
    metadata={"source": "pets_docs", "topic": "pets"}),
]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def basic_vector_store():
    # 3. Create a local Chroma vector store
    # vector_store = Chroma(
    #     collection_name="learning_collection",
    #     embedding_function=embeddings_model,
    #     persist_directory="./chroma_db",
    # )
    
    # # 4. Add documents
    # vector_store.add_documents(SAMPLE_DOCS)
    vector_store = Chroma.from_documents(
        documents=SAMPLE_DOCS, embedding=embeddings_model
    )

    # 5. Search for similar documents
    results = vector_store.similarity_search("What programming languages exist?", k=3)

    # 6. Display the results
    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Content: {result.page_content}")
        print(f"Metadata: {result.metadata}")
        print("-" * 40)

    # similarity_search_with_scores
    query = "What programming languages exist?"
    results_with_scores = vector_store.similarity_search_with_score(query, k=6)
    print(f"\nResults with scores for query '{query}':")    
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"Result {i+1}: {doc.page_content} (Score: {score})")

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available?"

        # without metadata filtering
        results = vectorstore.similarity_search(query, k=5)
        print(f"Results without metadata filtering for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        # with metadata filtering
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=5, filter=filter_criteria
        )
        print(f"\nResults with metadata filtering for query '{query}':")
        for i, doc in enumerate(filtered_results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )    

def persist_chroma():
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings_model,
        persist_directory=persist_dir,
    )

    original_count = vectorstore._collection.count()
    print(f"Persisted vector store with {original_count} documents.")
    print(f"Vector store persisted at: {persist_dir}")

    # simulate restart - load from disk
    del vectorstore

    reloaded = Chroma(
        embedding_function=embeddings_model,
        persist_directory=persist_dir,
    )

    reloaded_count = reloaded._collection.count()
    print(f"Reloaded vector store with {reloaded_count} documents.")

    # verify search still works
    results = reloaded.similarity_search("LangChain", k=2)
    print(f"Search result: {results[0].page_content[:50]}...")


def vector_store_setup():
    SAMPLE_DOCS = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
    ]

    sample_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50, chunk_overlap=10, separators=["\n", " ", ""]
    )

    chunks = sample_text_splitter.create_documents(SAMPLE_DOCS)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings_model, persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    results = retriever.invoke("How do I build AI applications?")
    print("Retriever results:")
    for i, doc in enumerate(results):
        print(
            f"Result {i+1}: {doc})"
        )

if __name__ == "__main__":
    # basic_vector_store()        
    # metadata_filtering()
    # persist_chroma()
    vector_store_setup()