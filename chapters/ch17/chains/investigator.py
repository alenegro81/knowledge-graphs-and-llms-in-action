import os
import json
import re
import uuid
from pathlib import Path
from typing import TypedDict

import json5
import neo4j
import yaml
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import jinja2_formatter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

import logging

from tools.schema import Neo4jSchema

logger = logging.getLogger("CHAIN")

# OLLAMA:
# from langchain_ollama import ChatOllama
# llm_model = ChatOllama(model="gemma3:12b", temperature=0.0)


# Openai:
# os.environ["OPENAI_API_KEY"] ="YOUR KEY"
# llm_model =  ChatOpenAI(model="gpt-4o", temperature=0)

# AzureOpenai:
from langchain_openai import AzureChatOpenAI

# os.environ["AZURE_OPENAI_API_KEY"] = "azure key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "endpoint"
# os.environ["AZURE_OPENAI_DEPLOYMENT"] = "Model deployment name"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
llm_model = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
                            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                            temperature=0)


class ChainConfiguration:
    def __init__(self):
        self.base = Path(__file__).parent
        self.config = self.load()

    def load(self):
        config_file = self.base / "chain_config.yaml"
        return yaml.load(config_file.open(), Loader=yaml.FullLoader)

    def get_prompt(self, name, **kwargs):
        system = self.config["prompts"][name].get("system")
        template_file = self.base / self.config["prompts"][name]["template"]
        template = template_file.read_text()
        prompt = jinja2_formatter(template, **kwargs)
        return system, prompt

    def getAnnotations(self, reload=True):
        if reload:
            self.config = self.load()
        return {"notes": self.config["notes"], "examples": self.config["examples"]}


class AgentState(TypedDict):
    # user question
    question: str
    # intent detection: response and reason
    output_type: str
    output_type_reason: str
    # schema description
    schema: str
    # text to cypher: generated query, reasoning and raw message
    query: str
    query_reasoning: str
    query_message: str
    # query execution errors
    results_error: list
    # summarization: summary, reasoning, analysis requested flag
    summary: str
    summary_reason: str
    summary_analisys: bool
    # query retrial: error information and execution count
    information: str
    retries: int


class Agent:
    def __init__(self, model, system=""):
        self.system = system
        self.neo4j_schema: Neo4jSchema = None
        self.memory = MemorySaver()
        self.selection = []
        self.results = None
        self.config = ChainConfiguration()
        graph = StateGraph(AgentState)
        graph.add_node("intent_detection", self.intent_detection)
        graph.add_edge("intent_detection", "schema_extraction")
        graph.add_node("schema_extraction", self.schema_extraction)
        graph.add_edge("schema_extraction", "text_to_cypher")
        graph.add_node("text_to_cypher", self.text_to_cypher)
        graph.add_edge("text_to_cypher", "query_execution")
        graph.add_node("query_execution", self.query_execution)
        graph.add_conditional_edges("query_execution",
                                    self.post_query_execution,
                                    {"retry": "text_to_cypher", "summarize": "generate_summary", "END": END})
        graph.add_node("generate_summary", self.generate_summary)
        graph.add_edge("generate_summary", END)
        graph.set_entry_point("intent_detection")
        self.graph = graph.compile(checkpointer=self.memory)
        self.model = model

    def run_prompt(self, prompt, system=""):
        messages = [HumanMessage(content=prompt)]
        if self.system or system:
            system = self.system if not system else system
            messages = [SystemMessage(content=system)] + messages

        message = self.model.invoke(messages)

        logger.debug(f" got {message.content}")
        payload = message.content
        payload = re.sub(r'^\s*```json\s*|\s*```\s*$', '', payload, flags=re.DOTALL)
        return json5.loads(payload)

    def intent_detection(self, state: AgentState):
        system, prompt = self.config.get_prompt("intent_detection", question=state["question"])
        results = self.run_prompt(prompt, system)
        return {"output_type": results["type"], "output_reason": results["reason"]}

    def schema_extraction(self, state: AgentState):
        assert self.neo4j_schema is not None, "you need to provide a neo4j schema"
        self.neo4j_schema.get_schema()
        self.neo4j_schema.apply_configuration()
        return {"schema": str(self.neo4j_schema), "retries": 0}

    def text_to_cypher(self, state: AgentState):
        extra = {
            "annotations": self.config.getAnnotations(),
            "selection": self.selection
        }
        params = dict(state) | extra
        system, prompt = self.config.get_prompt("text_to_cypher", **params)
        logger.debug(f"prompt: {prompt}")

        results = self.run_prompt(prompt, system)
        return {"query": results["query"], "query_reasoning": results["reasoning"],
                "query_message": json.dumps(results)}

    def query_execution(self, state: AgentState):
        try:
            results = self.neo4j_schema.run(state["query"])
            if state["output_type"] in {"graph", "map"}:
                self.results = list(results)
            else:
                self.results = results.to_df()
            results_error = None
            information = ""
        except neo4j.exceptions.ClientError as e:
            self.results = None
            results_error = str(e)
            logger.info(f"got error: {e}")
            information = f"""We tried:
                               {state['query']}
                               and we got:
                               ```
                               {str(e)}
                               ```"""
        retries = state.get("retries", 0) + 1
        return {"results_error": results_error, "retries": retries, "information": information}

    def post_query_execution(self, state: AgentState):
        # manage failure on the query execution
        if state["results_error"] is not None:
            if state["retries"] < 3:
                logger.info(f"{state['retries']} runs, we retry")
                return "retry"
            else:
                logger.info(f"{state['retries']} runs are enough")
                return "END"

        # summarize if map or graph, finish otherwise
        if state["output_type"] in ("map", "graph"):
            logger.info("summarizing..")
            return "summarize"
        else:
            logger.info("no summarization is needed")
            return "END"

    def generate_summary(self, state: AgentState):
        extra = {
            "records": self.results,
            "selection": self.selection
        }
        params = dict(state) | extra
        system, prompt = self.config.get_prompt("generate_summary", **params)
        logger.debug(prompt)
        results = self.run_prompt(prompt, system)
        return {"summary": results["summary"], "summary_reason": results["reasoning"],
                "summary_analisys": results["results_analysis"]}


abot = Agent(llm_model)


def set_schema(neo4j_schema):
    abot.neo4j_schema = neo4j_schema


def processQuestion(question, selection=None):
    config = {"recursion_limit": 40000, "configurable": {"thread_id": uuid.uuid4().hex}}
    if selection is not None:
        abot.selection = [{"label": list(node.labels)[0], "properties": dict(node)} for node in selection]
    else:
        abot.selection = []
    input = {"question": question}
    results = abot.graph.stream(input,
                                config=config,
                                stream_mode="updates")
    yield "update", "*detecting intent...*", input
    for result in results:
        node, value = list(result.items())[0]
        logger.info(f"got results: {node}, keys: {list(value.keys())}")
        current_state = abot.graph.get_state(config).values
        match node:
            case "intent_detection":
                yield "update", "*extracting schema...*", current_state
            case "schema_extraction":
                yield "update", "*generating query...*", current_state
            case "text_to_cypher":
                yield "update", "*executing the query...*", current_state
                yield "result", ("Reasoning", value["query_reasoning"]), current_state
            case "query_execution":
                if value["results_error"]:
                    yield "result", ("ERROR", value["results_error"]), current_state
                else:
                    output_type = current_state["output_type"]
                    yield output_type, abot.results, current_state
                    if output_type in {"graph", "map"}:
                        yield "update", "*summary generation...*", current_state
            case "generate_summary":
                yield "result", ("Summary", value["summary"]), current_state
    logger.info("no more results  sendin END")
    current_state = abot.graph.get_state(config).values
    yield "END", current_state, current_state
