import os
from pathlib import Path
import logging

import neo4j.graph
import folium
import yaml

from tools.schema import Neo4jSchema
import chains.investigator as chain

URI = os.getenv('NEO4J_URI', "neo4j://localhost:7687")
AUTH = (os.getenv('NEO4J_USER', "neo4j"), os.getenv('NEO4J_PASSWORD', 'password'))
DATABASE = os.getenv('NEO4J_DATABASE', "neo4j")

logger = logging.getLogger("APP")
logging.basicConfig(level=logging.INFO)

import streamlit as st
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle, Event
from streamlit_folium import st_folium

st.set_page_config(layout="wide")


class MessageHistory:
    def __init__(self):
        self.messages = [{}]

    def update(self, message, finalize=False):
        self.messages[-1].update(message)
        if finalize:  # we add an empty  message
            self.messages.append({})

    @staticmethod
    def display_message(msg):
        with st.chat_message("user"):
            st.markdown(msg["question"])
        with st.chat_message("assistant"):
            if "query_reasoning" in msg:
                st.markdown(f"##### Reasoning\n\n**output type**: `{msg['output_type']}`\n\n{msg['query_reasoning']}")
            if "table" in msg:
                st.table(msg["table"])
            if "map" in msg:
                map_ = folium.Map()
                nodes_to_map(msg["map"], map_)
                st_folium(map_)
            if "query" in msg:
                with st.expander("Query...", expanded=False):
                    st.markdown(f"```cypher\n\n{msg['query']}\n```")
                    st.json(msg["query_message"])
            if "summary" in msg:
                st.markdown(f"##### Summary\n\n{msg['summary']}")
                with st.expander("extra...", expanded=False):
                    st.json({
                        "summary_reason": msg["summary_reason"],
                        "summary_analisys": msg["summary_analisys"]
                    })
                    st.json(msg, expanded=False)

    def display_messages(self):
        for message in self.messages:
            if not message:
                continue
            self.display_message(message)


def setup_state():
    # initialize graph canvas
    if "canvas" not in st.session_state:
        st.session_state.canvas = {"elements": {}, "nodes": set(), "rels": set(), "byId": {}}
    chain.set_schema(Neo4jSchema(URI, AUTH, DATABASE))

    # ititalize graph selection
    if "selection" not in st.session_state:
        st.session_state.selection = set()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = MessageHistory()
        st.session_state.dbg = st.session_state.messages.messages

    # load configuration
    base = Path(__file__).parent
    config_file = base / "app.config.yaml"
    st.session_state.config = yaml.load(config_file.open(), Loader=yaml.FullLoader)

    return st.session_state


@st.cache_data
def node_styles():
    styles = []
    for nodel_lablel, properties in state.config["schema"]["node_styles"].items():
        styles.append(NodeStyle(nodel_lablel, properties["color"], properties["caption"], properties["icon"]))
    return styles


@st.cache_data
def edge_styles():
    return [EdgeStyle(label, caption='label', directed=True)
            for label in state.config["schema"]["relationship_styles"]]


def store_to_canvas(results):
    new_nodes: set[neo4j.graph.Node] = (
            {i for r in results for i in r if issubclass(type(i), neo4j.graph.Node)} |
            {n for r in results for i in r if issubclass(type(i), neo4j.graph.Path) for n in i.nodes})
    new_rels: set[neo4j.graph.Relationship] = (
            {i for r in results for i in r if issubclass(type(i), neo4j.graph.Relationship)} |
            {n for r in results for i in r if issubclass(type(i), neo4j.graph.Path) for n in i.relationships})

    state.canvas["byId"] |= {node.id: node for node in new_nodes} | {rel.id: rel for rel in new_rels}

    state.canvas["nodes"] = {item for item in state.canvas["byId"].values()
                             if issubclass(type(item), neo4j.graph.Node)}
    state.canvas["rels"] = {item for item in state.canvas["byId"].values()
                            if issubclass(type(item), neo4j.graph.Relationship)}

    state.canvas["elements"] = {
        "nodes": [{"data": dict(node) | {"id": node.id, "label": list(node.labels)[0]}}
                  for node in state.canvas["nodes"]],
        "edges": [{"data": {"id": i + len(state.canvas["nodes"]), "label": rel.type, "source": rel.nodes[0].id,
                            "target": rel.nodes[1].id}}
                  for i, rel in enumerate(state.canvas["rels"])]}

    return new_nodes, new_rels


def nodes_to_map(nodes, map):
    geo_nodes = {label: properties for label, properties in state.config["schema"]["node_styles"].items()
                 if "geo" in properties}
    bounds = None
    for node in nodes:
        # check if the node has a geo property
        properties = None
        for label in node.labels:
            if label in geo_nodes:
                geo_property = geo_nodes[label]["geo"]
                caption_property = geo_nodes[label]["caption"]
                properties = {
                    "geo": node[geo_property],
                    "caption": node[caption_property] if caption_property != "label" else label,
                    "color": geo_nodes[label].get("map_color", ""),
                    "icon": geo_nodes[label].get("map_icon", "")
                }
                break
        # if not skip continue with the next node
        if properties is None:
            continue
        # create the marker
        icon = folium.Icon(color=properties["color"], prefix="fa", icon=properties["icon"])
        folium.Marker(
            [properties["geo"].latitude, properties["geo"].longitude],
            popup=properties["caption"],
            tooltip=properties["caption"],
            icon=icon
        ).add_to(map)
        if bounds is None:
            bounds = [[properties["geo"].latitude, properties["geo"].longitude],
                      [properties["geo"].latitude, properties["geo"].longitude]]
        else:
            bounds = [[
                min(bounds[0][0], properties["geo"].latitude),
                min(bounds[0][1], properties["geo"].longitude)
            ], [
                max(bounds[1][0], properties["geo"].latitude),
                max(bounds[1][1], properties["geo"].longitude)
            ]]
    if bounds is not None:
        map.fit_bounds(bounds)


state = setup_state()

graph_column, chat_column = st.columns([2, 1])

with chat_column:
    st.markdown("## Chat")
    chat = st.container(height=900)
    with chat:
        # Display chat messages from history on app rerun
        state.messages.display_messages()

# React to user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    with chat:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
        selection = [state.canvas["byId"][int(item)] for item in state.selection]
        for response_type, response, current_state \
                in chain.processQuestion(question=str(question),
                                         selection=selection):
            logger.info(f"handling: {response_type}", )
            logger.info(f"updating keys: {list(current_state.keys())}")
            state.messages.update(current_state)
            match response_type:
                case "update":
                    # Display assistant response in chat message container
                    placeholder.markdown(response)
                case "graph" | "map":
                    # update the canvas with incoming nodes and relationships
                    placeholder.markdown("*updating canvas...*")
                    store_to_canvas(response)

                    # in case of map we add a mini map in the response
                    if response_type == "map":
                        state.messages.update({"map": state.canvas["nodes"]})
                case "table" | "chart":
                    state.messages.update({"table": response})
                    # Add assistant tabular response to chat
                    placeholder.table(response)
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                case "result":
                    title, content = response
                    response = f"##### {title}\n\n{content}"
                    # Add assistant response to chat
                    placeholder.write(response)
                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                case "END":
                    # Add assistant response to chat and finalize
                    logger.info(f"finalizing {list(current_state.keys())}")
                    state.messages.update(current_state, finalize=True)
                    st.rerun()

with graph_column:
    st.markdown("## Canvas")
    ret = st_link_analysis(state.canvas["elements"], "cose", node_styles(), edge_styles(), height=900,
                           events=[Event("add_to_selection", "dblclick dbltap", "node"), ])
    if ret is not None and ret["action"] == "add_to_selection":
        state["selection"].add(ret["data"]["target_id"])

with st.sidebar:
    st.markdown("## Selection")
    clear = st.button("clear selection")
    if clear:
        state["selection"].clear()
    for item in state["selection"]:
        selected_node = state.canvas["byId"][int(item)]
        ct, ci, cc = st.columns([1, 2, 1])
        with ct:
            st.write(f"{list(selected_node.labels)[0]} ({selected_node.id})")
        with ci:
            st.json(dict(selected_node), expanded=False)
        with cc:
            if st.button("remove", key=f"remove+{id(item)}"):
                state["selection"].remove(item)
                st.rerun()
