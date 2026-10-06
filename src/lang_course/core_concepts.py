from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# Demonstrates a basic chain using LCEL and Runnables

def demo_basic_chain():

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

# Demonstrates a batch chain using LCEL and Runnables.

def demo_batch_chain():
    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "Need short answer: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Execute the chain with a batch of inputs
    questions = [
        {"question": "What is FastAPI?"},
        {"question": "What is LangGraph?"},
        {"question": "In LangChain, what does LCEL stand for?"}
    ]
    results = chain.batch(questions)
    
    for i, result in enumerate(results):
        print(f"Response to question {i+1}: {result}")

    return chain

# Demonstrate streaming for real-time output.    

def demo_streaming():
    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming

# Demonstrate input/output schema inspection.

def demo_schema_inspection():
    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def exercise_first_chain():
    
    # EXERCISE: Create a chain that:
    # 1. Takes a product name and target audience
    # 2. Generates a marketing tagline
    # 3. Returns just the tagline as a string

    # Test with: product="AI Course", audience="developers"

    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Tagline: {result}")    

def new_way():
    # the univeral way to initialize a model
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)

    # Or provider-specific (still works)

    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    openai_model = ChatOpenAI(model="gpt-4o-mini",
                              temperature=0.7,
                              max_tokens=1500,
                              timeout=30,
                              max_retries=3)    


def main():
    # print("======BASIC CHAIN======")
    # demo_basic_chain()        
    # print("======BATCH CHAIN======")
    # demo_batch_chain()
    # print("======STREAMING CHAIN======")
    # demo_streaming()
    # print("======SCHEMA INSPECTION======")
    # demo_schema_inspection()
    print("======EXERCISE: FIRST CHAIN======")
    exercise_first_chain()


if __name__ == "__main__":
    main()