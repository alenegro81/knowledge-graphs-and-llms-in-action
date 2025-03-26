import networkx as nx


def graph_from_cypher(results):
    G = nx.MultiDiGraph()
    nodes = list(results._nodes.values())
    for node in nodes:
        G.add_node(node.id, **{**{**node._properties}, **{"labels": list(node._labels)}})

    rels = list(results._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id,
                   **{**{"key": rel.id}, **{"label": rel.type}, **{**rel._properties}})
    return G


def graph_undirected_from_cypher(results):
    G = nx.Graph()
    nodes = list(results._nodes.values())
    for node in nodes:
        G.add_node(node.id, **{**{**node._properties}, **{"labels": list(node._labels)}})

    rels = list(results._relationships.values())
    for rel in rels:
        G.add_edge(rel.start_node.id, rel.end_node.id,
                   **{**{"key": rel.id}, **{"label": rel.type}, **{**rel._properties}})
    return G
