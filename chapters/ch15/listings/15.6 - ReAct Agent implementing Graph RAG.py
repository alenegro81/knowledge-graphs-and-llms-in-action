import os
import sys
from pprint import pprint

from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.agents import create_structured_chat_agent, Tool, AgentExecutor

# Add the `code` directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
from definitions import KG_SCHEMA
from tools import vector_search, kg_reader, kg_doc_selector, VectorSearchInput, KGReaderInput, REDiarySelectorInput

from dotenv import load_dotenv
load_dotenv()

MODEL = "gpt-4o"

def load_custom_prompt(path: str):
    with open(path, 'r') as f:
        system = f.read()
        human = ("{input}\n\n"
                 "{agent_scratchpad}\n\n"
                 "(reminder to respond in a JSON blob no matter what)")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", human),
            ]
        )
    return prompt

tools = [
        StructuredTool.from_function(
            #func=vector_qa.invoke,
            func=vector_search,
            name="Diaries-vector-search",
            args_schema=VectorSearchInput,
            return_direct=False,
            description="""This is a backup tool based on vector search to be used ONLY when no other tool provides nor  
            is capable to provide question-relevant context.
            Try the other tools first! Then use this one as a last resort.
            Use it when you need to answer questions about details within Warren Weaver's diaries that are too
            fine-grained to be modeled in the Knowledge Graph.
            When the other tools return nothing useful, execute this tool before generating final answer.
            Always use full question as input, without any changes!""",
        ),
        StructuredTool.from_function(
            func=kg_reader,
            name="KnowledgeGraph-reader",
            args_schema=KGReaderInput,
            description=f"""Useful when you need to answer questions for which the information stored in the KG
            is sufficient, for example about relationships among entities such as people, organizations and occupations.
            Also useful for any sort of aggregation like counting the number of people per occupation etc.
            Use it also as an intermediate step to obtain structured information needed as an input to another tool.
            This tool translates the question into a Cypher query, executes it and returns results.

            Full Knowledge Graph schema in Cypher syntax to help you decide whether this tool can be used or not:
            {KG_SCHEMA}

            Always use full question as input, without any changes!""",
        ),
        StructuredTool.from_function(
            func=kg_doc_selector,
            name="KG-based-document-selector",
            args_schema=REDiarySelectorInput,
            return_direct=False,
            description=(
                "Use this tool when the question asks for detailed information regarding interaction between two "
                "entities. It returns highly-relevant documents that specifically mention the kind of entity-entity "
                "relationship the question is about.\n"
                "The entities and relationship between them must be modeled within the KG (see schema below), but the KG itself "
                "does not contain enough details to provide the answer (if it does, you must use the "
                "KnowledgeGraph-reader tool.\n"
                "Use the KnowledgeGraph-reader tool first if you need to obtain concrete list of entities.\n\n"
                "Full Knowledge Graph schema in Cypher syntax to help you decide whether this tool can be used or not:\n"
                f"{KG_SCHEMA}"
            )
        )
    ]

# initialise basic LLM model
llm = ChatOpenAI(model=MODEL, temperature=0)

# load prompt
#prompt = hub.pull("hwchase17/structured-chat-agent")
prompt = load_custom_prompt("../prompts/prompt_structured.txt")

# create structured ReAct agent & its executor
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, return_intermediate_steps=True, verbose=True)

if __name__ == "__main__":
    question = "What did her fellow researchers say about Dorothy M. Wrinch?"
    #question = "Are there any shared research topics between Harvard University and Johns Hopkins University?"

    response = agent_executor.invoke({"input": question})

    pprint(f"\n### Agent's intermediate steps:\n")
    for i, step in enumerate(response['intermediate_steps']):
        print(f"\n--- Step {i}")
        for j, m in enumerate(step[0].messages):
            print(m.pretty_repr())
    print(f"\n### Agent's response: {response['output']}")