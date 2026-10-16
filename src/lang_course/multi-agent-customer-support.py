"""A customer sends a message to an AI customer support system. A supervisor decides which specialist should handle it.
For example:
- "I was charged twice for my subscription." → Billing Agent
- "I cannot log into my account." → Technical Support Agent
- "How can I change my email address?" → General Support Agent"""

from typing import Literal
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# Define the shared state
class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_complete: bool
    final_response: str
    agent_calls: int

# Define the supervisor's decision schema
class SupportDecision(BaseModel):
    next: Literal[
        "billing",
        "technical",
        "general",
        "FINISH",
    ] = Field(description="Which agent should act next")

    reasoning: str = Field(
        description="Why this agent was selected"
    )

# Create the supervisor
supervisor_llm = llm.with_structured_output(SupportDecision)

MAX_AGENT_CALLS = 3

def supervisor(state: SupportState) -> dict:
    print("11" * 50)

    if state["agent_calls"] >= MAX_AGENT_CALLS:
        return {
            "task_complete": True,
            "next_agent": "FINISH",
        }

    system_prompt = """
    You are the supervisor of a customer support system.

    You manage three specialist agents:

    1. billing:
       Handles duplicate charges, payments,
       subscriptions, refunds, and invoices.

    2. technical:
       Handles login problems, bugs, errors,
       and technical issues.

    3. general:
       Handles account information, changing email
       addresses, and general customer questions.

    Choose the most appropriate specialist based
    on the customer's message.

    If a specialist has already answered the customer's
    question adequately, choose FINISH.

    Return your decision and explain your reasoning.
    """

    decision = supervisor_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )
    
    if decision.next == "FINISH":
        return {
            "task_complete": True,
            "next_agent": "FINISH",
        }

    return {
        "next_agent": decision.next,
        "messages": [
            AIMessage(
                content=(
                    f"[Supervisor] Routing to "
                    f"{decision.next} agent. "
                    f"Reason: {decision.reasoning}"
                )
            )
        ],
    }


# Create the billing specialist
def billing_agent(state: SupportState) -> dict:
    print("22" * 50)
    response = llm.invoke(
        [
            SystemMessage(
                content="""
                You are a billing support specialist.
                Help customers with duplicate charges,
                subscriptions, refunds, and payments.
                Be polite and practical.
                Never claim that a refund was processed
                unless you have confirmation.
                """
            ),
            *state["messages"],
        ]
    )

    return {
    "messages": [response],
    "agent_calls": state["agent_calls"] + 1,
}

# Create the technical specialist
def technical_agent(state: SupportState) -> dict:
    print("33" * 50)
    response = llm.invoke(
        [
            SystemMessage(
                content="""
                You are a technical support specialist.
                Help customers with login problems,
                bugs, errors, and technical issues.
                Give clear, step-by-step troubleshooting.
                Ask for more details if necessary.
                """
            ),
            *state["messages"],
        ]
    )

    return {
    "messages": [response],
    "agent_calls": state["agent_calls"] + 1,
}


# Create the general support specialist
def general_agent(state: SupportState) -> dict:
    print("44" * 50)
    response = llm.invoke(
        [
            SystemMessage(
                content="""
                You are a general customer support specialist.
                Help customers with account information,
                changing email addresses, and general queries.
                Give friendly and easy-to-follow instructions.
                """
            ),
            *state["messages"],
        ]
    )

    return {
    "messages": [response],
    "agent_calls": state["agent_calls"] + 1,
}


# Create the finalizer
def finalize(state: SupportState) -> dict:
    print("55" * 50)
    for message in reversed(state["messages"]):
        if (
            isinstance(message, AIMessage)
            and not message.content.startswith("[Supervisor]")
        ):
            return {"final_response": message.content}

    return {
        "final_response": "Sorry, I couldn't generate a response."
    }  

# Define routing
def route_to_agent(state: SupportState) -> str:
    print("66" * 50)
    if state["task_complete"]:
        return "finalize"

    return state["next_agent"]   

# Build the graph
def create_support_system():
    graph = StateGraph(SupportState)
    print("77" * 50)
    graph.add_node("supervisor", supervisor)
    graph.add_node("billing", billing_agent)
    graph.add_node("technical", technical_agent)
    graph.add_node("general", general_agent)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "billing": "billing",
            "technical": "technical",
            "general": "general",
            "finalize": "finalize",
        },
    )

    graph.add_edge("billing", "supervisor")
    graph.add_edge("technical", "supervisor")
    graph.add_edge("general", "supervisor")

    graph.add_edge("finalize", END)

    return graph.compile()    


if __name__ == "__main__":
    support = create_support_system()
    result = support.invoke({
        "messages": [
            HumanMessage(
                content="I was charged twice for my subscription."
            )
        ],
        "next_agent": "",
        "task_complete": False,
        "final_response": "",
        "agent_calls": 0,
    })

    print("Final response:")
    print(result["final_response"])
