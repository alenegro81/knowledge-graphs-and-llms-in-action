from dataclasses import dataclass
from pathlib import Path

import yaml
from neo4j import GraphDatabase


@dataclass
class Property:
    """represents a node or relationship property with an optional description"""
    name: str
    type: str
    description: str = None

    def __str__(self):
        """represents the property as string in the format: propety_name:TYPE /* optional description */"""
        ret = f"{self.name}: {self.type}"
        if self.description is not None:
            ret += f" /* {self.description} */"
        return ret


@dataclass
class Node:
    """Represents a node type."""
    items = {}
    name: str
    properties: list[Property]
    description: str = None

    @classmethod
    def mk_node(cls, name, value):
        """Creates a new node with the given name and properties from a dictionary.

        Args:
            name (str): The name of the node.
            value (dict): the node description as returned by `apoc.meta.schema`
        """
        properties = [Property(name=k, type=v["type"]) for k, v in value["properties"].items()]
        properties = sorted(properties, key=lambda x: x.name)
        node = Node(name=name,
                    properties=properties)
        for rel_name, rel_value in value["relationships"].items():
            Relationship.mk_rels(source=name, name=rel_name, value=rel_value)

        cls.items[node.name] = node

    def drop_properties(self, skipProperties):
        """Drops specified properties from the node.

        Args:
            skipProperties (list): A list of property names to be dropped.
        """
        self.properties = [prop for prop in self.properties if prop.name not in skipProperties]

    def __str__(self):
        """represents the node as string in the format:
           (:NodeType /* node class description */ {
              property_one:TYPE /* property one description */,
              property_two:TYPE /* property two description */,
              ...
           })
        """
        descr = "" if self.description is None else f"/* {self.description} */ "
        return (
                f"(:{self.name} {descr}{{\n    " +
                ",\n    ".join(str(prop) for prop in self.properties) +
                "\n})\n"
        )


@dataclass
class Relationship:
    """Represents a relationship type between two node types."""
    name: str
    source: str
    dest: str
    properties: list[Property]
    description: str = None
    items = {}

    @classmethod
    def mk_rels(cls, source, name, value):
        """Creates relationships between nodes based on the given parameters.

        Args:
            source (str): The source node type.
            name (str): The relationship name.
            value (dict): The relationship description as returned by `apoc.meta.schema`
        """
        if value["direction"] != "out":
            return
        for dest in value["labels"]:
            properties = [Property(name=k, type=v["type"]) for k, v in value["properties"].items()]
            properties = sorted(properties, key=lambda x: x.name)
            rel = Relationship(name, source, dest, properties)
            cls.items[f"({source})-[{name}]->({dest})"] = rel

    def drop_properties(self, skipProperties):
        """Drops specified properties from the relationship.

        Args:
            skipProperties (list): A list of property names to be dropped.
        """
        self.properties = [prop for prop in self.properties if prop.name not in skipProperties]

    def __str__(self):
        """Represents the relationship as string in the format:
        (:Source)-[:RELATIONSHIP {prop_1:TYPE,...}]->(:Destination)."""
        properties = ", ".join(str(prop) for prop in self.properties)
        return f"(:{self.source})-[:{self.name} {{{properties}}}]->(:{self.dest})"


class Neo4jSchema:
    def __init__(self, uri, auth, database):
        self.driver = GraphDatabase.driver(uri, auth=auth, database=database)

    def run(self, query, **args):
        return self.driver.session().run(query, parameters=args)

    def close(self):
        self.driver.close()

    def get_schema(self):
        with self.driver.session() as session:
            result = list(session.run("CALL apoc.meta.schema({sample:-1}) "))[0]["value"]
        [Node.mk_node(k, v) for k, v in result.items() if v["type"] == "node"]

    @staticmethod
    def apply_configuration(config: dict = None):
        if config is None:
            config_file = Path(__file__).parent / "schema_config.yaml"
            config = yaml.load(config_file.open(), Loader=yaml.FullLoader)["schema"]

        # filter nodes by classes
        items = {node.name: node for node in Node.items.values() if node.name not in config["skip"]["classes"]}
        Node.items = items

        # filter classes properties
        for node in Node.items.values():
            node.drop_properties(config["skip"]["properties"])

        # apply descriptions to nodes
        for node in Node.items.values():
            # node description from configuration (None if not specified)
            node.description = config["descriptions"]["classes"].get(node.name)
            for prop in node.properties:
                # get property description for this class.property (None otherwise)
                property_description = config["descriptions"]["properties"].get(node.name, {}).get(prop.name)
                prop.description = property_description

        # filter relationships
        relationships = {rel_name: rel
                         for rel_name, rel in Relationship.items.items()
                         if rel.source not in config["skip"]["classes"]  # source is not skip_class
                         if rel.dest not in config["skip"]["classes"]  # dest is not skip_class
                         if rel.name not in config["skip"]["relationships"]  # relation is not in skip_relationship
                         }

        # filter relationships properties
        for rel in Relationship.items.values():
            rel.drop_properties(config["skip"]["properties"])

        Relationship.items = relationships

    def __str__(self):
        # set headers
        ret = ["### Graph Schema Overview\n",
               "#### Node Types"]
        # append node representations
        ret += [str(node) for node in Node.items.values()]

        ret.append("#### Relationships\n")

        # append relationship representations
        ret += [str(rel) for rel in Relationship.items.values()]

        return "\n".join(ret)
