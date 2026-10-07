from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate
)
from langchain_core.runnables import (
    RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch)

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0)

def basic_chain():
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke(
        {
            "text": "LangChain is a framework for developing applications powered by language models."
        }
    )
    print(f"Summary: {result}  ")


def parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in two sentences: {text}"
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract 5 keywords in the following text: {text}\nReturn as a comma-separated list."
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "What is the sentiment of the following text? {text}"
    )

    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | StrOutputParser(),
        keywords=keywords_prompt | model | StrOutputParser(),
        sentiment=sentiment_prompt | model | StrOutputParser(),
    )

    text = """
    LangChain is a framework for developing applications powered by language models. It provides a standard interface for all LLMs, as well as a toolkit of components that can be easily assembled into a variety of applications, ranging from chatbots to Generative Question-Answering (GQA) to summarization tools.
    """

    result = analysis_chain.invoke({"text": text})
    print("Analysis Result:")
    print(result)
    print("==================================")
    for key, value in result.items():
        print(f"  {key.capitalize()}: {value}")


def passthrough_chain():
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n\n"
        "Answer the question based on the context."
    )

    parser = StrOutputParser()

    def fake_retriever(input_dict):
        # Simulate retrieving context based on the question
        question = input_dict["question"]

        return "LangChain was created by Harrison Chase in 2022."

    x1 = RunnableParallel(
        context=RunnableLambda(fake_retriever),
        question=RunnablePassthrough()
    )

    x2 = RunnableLambda(
        lambda x: {
            "context": x["context"],
            "question": x["question"]["question"]
        }
    )

    chain = x1 | x2 | prompt | model | parser

    result = chain.invoke({"question": "Who created LangChain?"})

    print(f"Answer: {result}")

def chain_branching():
    general_prompt = ChatPromptTemplate.from_template("Answer to this question:{question}")
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {question}"
    )

    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {question}\nReturn only the classification.")

    classifier_chain = classifier_prompt | model | StrOutputParser()

    def is_code_question(input_dict):
        classification = classifier_chain.invoke(input_dict)
        return "code" in classification.lower()

    branching_chain = RunnableBranch(
       ( is_code_question, code_prompt | model | StrOutputParser())
       , general_prompt | model | StrOutputParser()
    )

    results = branching_chain.invoke({"question": "Is today rainy?"})
    print(f"Branching Result: {results}")

def demo_debbuging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Get configuration
    print("Chain input schema:", chain.input_schema.model_json_schema())
    print("Chain output schema:", chain.output_schema.model_json_schema())    

    # Method 2: Use with_config for tacing
    result = chain.with_config(
        run_name="greeting_chain",
        # tags="demo,debugging",
    ).invoke({"name": "Alice"})
    print(f"Greeting: {result}")

    # Method 3: Inspect intermediate steps
    # Using RunnableLambda for logging
    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )

    print("\nDebug chain execution:")
    result = debug_chain.invoke({"name": "Debug"})
    print(f"Greeting: {result}")


if __name__ == "__main__":
    # basic_chain()
    # parallel_chain()
    # passthrough_chain()
    # chain_branching()
    demo_debbuging()