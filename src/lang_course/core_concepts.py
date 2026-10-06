from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# Demonstrates a basic chain using LCEL and Runnables
def demo_basic_chain():
        """Demonstrates a basic chain using LCEL and Runnables."""

        # Component 1: Define the prompt template using LCEL
        prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
        )
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        parser = StrOutputParser()

        # Compose with pipe operator
        chain = prompt | model | parser

        # Execute the chain with an input
        result = chain.invoke({"question": "What is LangChain?"})
        print(f"Response: {result}")

        return chain

def main():
    print("======BASIC CHAIN======")
    demo_basic_chain()        

if __name__ == "__main__":
    main()