from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import add_messages
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

#A simple graph
class SimpleState(TypedDict):
    input: str
    output: str
    step: int

def simple_graph():
    def process(state: SimpleState) -> dict:
        return {
            "output": state["input"].upper(),
            "step": state["step"] + 1
        }
    
    graph = StateGraph(SimpleState)
    graph.add_node("process",process)
    graph.add_edge(START,"process")
    graph.add_edge("process",END)

    app = graph.compile()
    result = app.invoke(
        {
            "input": "sukanya",
            "output":"",
            "step":0
        }
    )

    print(result)

#State with reducers
class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add]
    count: Annotated[int, operator.add]    

def accumulating_state():
    def step_one(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 1 executed"],
            "count": 1
        }

    def step_two(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 2 executed"],
            "count": 1
        }
    
    graph = StateGraph(AccumulatingState)
    graph.add_node("step_one",step_one)
    graph.add_node("step_two",step_two)
    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two")
    graph.add_edge("step_two", END)

    app = graph.compile()
    results = app.invoke({"messages": ["Testing states"], "count": 0})

    print(results)


#Message state with an LLM
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = init_chat_model("gpt-4o-mini", temperature=0)

def message_state():
    def chat_node(state: MessageState) -> dict:
          response = llm.invoke(state["messages"])
          return {"messages": [response]}
    
    graph = StateGraph(MessageState)
    graph.add_node("chat_node",chat_node)
    graph.add_edge(START,"chat_node")
    graph.add_edge("chat_node",END)

    app = graph.compile()
    result = app.invoke({
    "messages": [
        HumanMessage(content="Say Hello in Tagalog")
    ]
    })

    # print(result)
    for m in result["messages"]:
       m.pretty_print()


# A multi-node AI graph       
class MultiStepState(TypedDict):
    input: str
    analyzed: str
    enhanced: str
    final: str

def multi_node_state():
    def analyze_node(state: MultiStepState) -> dict:
        response = llm.invoke([
            HumanMessage(
                content=(
                    "Analyze the following input and summarize it in "
                    f"one sentence: {state['input']}"
                )
            )
        ])
        return {"analyzed": response.content}

    def enhance(state: MultiStepState) -> dict:
        response = llm.invoke([
            HumanMessage(
                content=(
                    "Take the following analysis and enhance it "
                    f"with more details: {state['analyzed']}"
                )
            )
        ])
        return {"enhanced": response.content}    

    def finalize(state: MultiStepState) -> dict:
        response = llm.invoke([
            HumanMessage(
                content=(
                    "Take the following enhanced analysis and finalize "
                    f"it into a concise summary: {state['enhanced']}"
                )
            )
        ])
        return {"final": response.content}

    
    graph = StateGraph(MultiStepState)
    graph.add_node("analyze_node",analyze_node)
    graph.add_node("enhance",enhance)
    graph.add_node("finalize",finalize)

    graph.add_edge(START,"analyze_node")
    graph.add_edge("analyze_node","enhance")
    graph.add_edge("enhance","finalize")
    graph.add_edge("finalize",END)

    app = graph.compile()

    result = app.invoke({
    "input": "Artificial intelligence"
    })  

    # print(result)  
    print("\nMulti-Node Graph Result:")
    print(f"  Input: {result['input']}")
    print(f"  Analyzed: {result['analyzed'][:100]}...")
    print(f"  Enhanced: {result['enhanced'][:100]}...")
    print(f"  Final: {result['final']}")



"""
EXERCISE: Create a LangGraph that:
1. Takes a topic as input
2. Node 1: Generates 3 questions about the topic
3. Node 2: Answers one of the questions
4. Returns both questions and answer
"""

class QAState(TypedDict):
    topic:str
    questions:str
    answer:str   


def qa_langgraph():
    llm = init_chat_model("gpt-4o-mini", temperature=0)
    
    def gen_qns(state:QAState) -> dict:
          response = llm.invoke(
                    f"Analyze the given input and generate three questions about the topic: {state['topic']}\n"
                    "Format: numbered list"
                )
            
          return {"questions": response.content}

    def ans_qns(state:QAState) -> dict:
        response = llm.invoke(
            f"Answer the second question fom the list: \n{state['questions']}"
        )

        return {"answer": response.content}

    graph = StateGraph(QAState)
    graph.add_node("gen_qns",gen_qns)
    graph.add_node("ans_qns",ans_qns)

    graph.add_edge(START,"gen_qns")
    graph.add_edge("gen_qns","ans_qns")
    graph.add_edge("ans_qns",END)

    app = graph.compile()

    result = app.invoke({
    "topic": "Angular"
    })  

    print("\nQA Graph Result:")
    print(f"  Topic: {result['topic']}")
    print(f"  Questions: {result['questions']}...")   
    print(f"  Answer: {result['answer']}...")

if __name__ == "__main__":
    # simple_graph()
    #  accumulating_state()
    # message_state()
    # multi_node_state()
    qa_langgraph()