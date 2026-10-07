from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


load_dotenv()


parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("write a short poem about {topic}")
model = init_chat_model(model="gpt-4o-mini", temperature=0)
chain = prompt | model | parser

# response = chain.invoke({"topic": "nature"})
# print(response)  # This will print the generated poem as a string
# print("==================================")
# print(type(response))


parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name' and 'age' for: {description}"
)

chain = prompt | model | parser

# result = chain.invoke({"description": "A 25-year-old developer named Alex"})
# print(result)


class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template(
    "Return a JSON object with 'name', 'age', and 'occupation' for: {description}"
).partial(format_instructions=parser.get_format_instructions())


chain = prompt | model | parser

# result = chain.invoke({"description": "A 30-year-old software engineer named Sam"})
# print(result)


# Structured Output
class StructuredOutput(BaseModel):
    title: str = Field(description="The title of the output")
    content: str = Field(description="The main content of the output")
    tags: list[str] = Field(description="A list of tags associated with the output")

structured_model = model.with_structured_output(StructuredOutput)
# result = structured_model.invoke("Write a blog post about the benefits of AI in healthcare.")
# print(result)


class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year of the movie")
    director: str = Field(description="The director of the movie")
    actors: list[str] = Field(description="A list of main actors in the movie")
    genre: str = Field(description="The genre of the movie")
    rating: float = Field(description="The movie's rating out of 10")

def movie_parser(review: str) -> Movie:

    prompt = ChatPromptTemplate.from_template(
        "Extract movie information from this review:\n\n{review}"
    )
    model = init_chat_model(model="gpt-4o-mini", temperature=0)
    structured_model = model.with_structured_output(Movie)

    chain = prompt | structured_model

    result = chain.invoke({"review": review})

    print(result)

movie_parser("The Dark Knight (2008) directed by Christopher Nolan is an absolute masterpiece.Christian Bale and Heath Ledger deliver incredible performances in this action thriller. 10/10!")   


