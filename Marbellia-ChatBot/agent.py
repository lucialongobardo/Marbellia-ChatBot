from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils import get_question_context, google_search_result
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')

# Definimos el template para la consulta de turismo
turism_template = """You are a very experienced turist guide specialised in recommending activities \
and things to do in Marbella, a city located in Andalusia, Spain. \
You have an excellent knowledge of and understanding of restaurants, sports, activities, experiences and places to visit in the city \
specifically targeted to families, couples, friends and solo travelers. \
You have the ability to think, reflect, debate, discuss and evaluate the data stored in a knowledge base from youtube videos related to \
turism in Marbella, and the ability to make use of it to support your explanations to the future turists that will visit the city and ask for your advice. \
Remenber: You answer must be so accurate and based on your knowledbase. \
Here is a question from a user: \
{input}"""

default_template = """You are a bot specialised in giving answers to questions about a wide range of topics. \
You are provided with the user answer and context from the first non-sponsored URL from a Google search. \
If you don't know the answer simply say I don't know but if you do please answer the question precisely.\
Here is a question from a user and a bit of context from Google Search: \
{input}"""

def get_turism_answer(input):
    input = get_question_context(query=input, top_k=3)
    llm_prompt = PromptTemplate.from_template(turism_template)
    chain = LLMChain(llm=llm, prompt=llm_prompt)
    answer = chain.run(input)
    return answer

def get_internet_answer(input):
    context = google_search_result(input)
    input = f"Pregunta del usuario: {input} \n Contexto para responder a la pregunta del usuario: {context}"
    llm_prompt = PromptTemplate.from_template(default_template)
    chain = LLMChain(llm=llm, prompt=llm_prompt)
    answer = chain.run(input)
    return answer

tools = [
    Tool(
        name='Turism knowledgebase tool',
        func=get_turism_answer,
        description=('Use this tool when answering questions about turism in Marbella.')
    ),
        Tool(
        name='Default knowledgebase tool',
        func=get_internet_answer,
        description=(
            'use this tool when the input question is not related to turism in Marbella.'
        )
    )
]

llm = ChatOpenAI(model='gpt-4',temperature=0)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

def call_agent(input):
    return agent(input)['output']