from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

def demo_init_chat_model():

    chat_model = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.7,
        streaming=True,
        max_retries=3,
    )

    response = chat_model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response: {response.content}")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        ),
    }

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)  # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")    

    # EXERCISE: Create a function that:
    # 1. Takes a question and a list of model names
    # 2. Gets responses from all models
    # 3. Returns a dict of {model_name: response}

    # Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]

def compare_models(question, model_names):
    responses = {}
    for model_name in model_names:
        model = init_chat_model(
            model=model_name,
            temperature=0.7,
            streaming=False,
        )
        response = model.invoke(question)
        responses[model_name] = response.content
    return responses    


def compare_llm():
    # Test
    result = compare_models(
        question="What is AI?",
        model_names=["gpt-4o-mini", "gpt-4o"]
    )

    print(result)    
    for model, answer in result.items():
        print(f"Response from {model}: {answer}\n")
    


if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    # demo_message()
    compare_llm()