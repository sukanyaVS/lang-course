from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv() 



class ResearchResponse(BaseModel):
    answer: str
    confidence: str
    sources: List[str]
    key_quotes: List[str] = []
    follow_up_questions: List[str]


class AIResearchAssistant:
    def __init__(
        self,
        persist_directory: str = "./research_demo_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
      self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

      self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

      self.text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size, chunk_overlap=chunk_overlap,
          separators=["\n\n", "\n", " ", ""]
      )

      self.persist_directory = persist_directory

      self.vectorStore = Chroma(
          persist_directory=persist_directory,
          embedding_function=self.embeddings,
          collection_name="research_docs"
      )

      self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

      self.parser = StrOutputParser()


      print(f"Initialized AIResearchAssistant with persist_directory: {persist_directory}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
      print(f"Document store initialized with {self.vectorStore._collection.count()} documents.")

    def add_docs(self, docs: List[Document], source_name: Optional[str] = None):
        if source_name:
            for doc in docs:
                doc.metadata["source"] = source_name

        chunks = self.text_splitter.split_documents(docs)        
        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata["indexed_at"] = datetime.now().isoformat()

        self.vectorStore.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {len(docs)} documents")
        return len(chunks)

    def add_text(self, text: str, source: str, metadata: dict = None):
        if metadata is None:
            metadata = {}
        metadata["source"] = source
        doc = Document(page_content=text, metadata=metadata)
        return self.add_docs([doc], source_name=source)

    def add_texts(self, texts: List[str], source: str):
        docs = [Document(page_content=text, metadata={"source": source}) for text in texts]
        return self.add_docs(docs, source_name=source)

    def get_document_count(self):
        return self.vectorStore._collection.count()
    
    def list_sources(self):
        results = self.vectorStore._collection.get()
        sources = set()
        for metadata in results.get("metadatas", []):
          if metadata and "source" in metadata:
            sources.add(metadata["source"])

        return sorted(list(sources))
      
    def create_retriever(self, use_advanced:bool = False):
        base_retriever = self.vectorStore.as_retriever(
            search_type = "similarity", search_kwargs= {"k":4}        
            )

        if not use_advanced:
            return base_retriever

        multi_retriever = MultiQueryRetriever.from_llm(
           retriever = base_retriever,
           llm = self.llm
        )

        return multi_retriever

    def _format_docs_for_context(self, docs) -> str:
        if not docs:
            return "No relevant documents found."

        formatted = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(
                f"[Source {i+1}: {source}]\n{doc.page_content}"
            )

        return "\n\n--\n\n".join(formatted)    

    def get_session_history(self, session_id:str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
          self.session_store[session_id] = InMemoryChatMessageHistory()

        return self.session_store[session_id]  

    def ask(self,question:str,session_id:str,use_advanced:bool = True):
        history = self.get_session_history(session_id)
        retriever = self.create_retriever(use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI Research Assistant.ONLY use information from the provided context"),
            MessagesPlaceholder(variable_name="history"),
           ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
        )

        chain = prompt | self.llm | self.parser

        response = chain.invoke({"context":context, "question":question, "history": history.messages[-10:]})

        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))

        return response

    def ask_structured(self,question:str,session_id:str,use_advanced:bool = True):
        history = self.get_session_history(session_id)
        retriever = self.create_retriever(use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)
        sources = list(set(d.metadata.get("source", "Unknown") for d in docs))

        structured_llm = self.llm.with_structured_output(
          ResearchResponse
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI Research Assistant.ONLY use information from the provided context"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ]
        )

        chain = prompt | structured_llm

        response = chain.invoke(
        {
            "context": context,
            "question": question,
            "sources": ", ".join(sources),
            "history": history.messages[-10:],
        }
        )

        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response.answer))
        return response

    def clear_session(self, session_id: str):
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")

    def get_session_messages(self, session_id: str) -> list:
        if session_id not in self.session_store:
            return []

        return [
            {
                "role": "human"
                if isinstance(m, HumanMessage)
                else "assistant",
                "content": m.content,
            }
            for m in self.session_store[session_id].messages
        ]        


if __name__ == "__main__":
    assistant = AIResearchAssistant()
    
    # Example usage:
    docs = [Document(page_content="Sample content. This is a simple example. Testing the assistant.", metadata={"source": "example"})]
    assistant.add_docs(docs, source_name="example_source")

    assistant.add_text(
    text="RAG retrieves relevant information before generating an answer.",
    source="rag_notes.txt",
    )

    assistant.add_texts(
    [
        "RAG combines retrieval and generation.",
        "Embeddings represent text as vectors.",
        "Chroma stores vectors for similarity search.",
    ],
    source="my_research_notes",
    )

    print("Total count", assistant.get_document_count())
    print("Sources",assistant.list_sources())

    retriever = assistant.create_retriever()
    docs = retriever.invoke("What is RAG")
    
    print(assistant._format_docs_for_context(docs))

    question = "What is RAG?"

    result = assistant.ask(question, "user_A")
    print("--" * 60)
    print(result)

    print("--" * 30)
    results = assistant.ask_structured(question,"user_A")
    print(results)

    result = assistant.ask(question, "user_B")
    print("*****" * 60)
    print(result)

    print("BEFORE CLEARING")
    print("*USER A**--------------")
    print(assistant.get_session_messages("user_A"))

    print("*USER B**---------------")
    print(assistant.get_session_messages("user_B"))

    assistant.clear_session("user_A")
    print("AFTER CLEARING")

    print("*USER A**--------------")
    print(assistant.get_session_messages("user_A"))

    print("*USER B**---------------")
    print(assistant.get_session_messages("user_B"))