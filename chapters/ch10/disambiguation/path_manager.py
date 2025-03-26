from tqdm import tqdm
import itertools
import json
from utils import minify_json, minify_text
from logger import Logger


class PathExtraction():
  def __init__(self, model, store, candidates, logger=None):
    self.model = model
    self.store = store
    self.candidates = candidates
    self.logger = logger if logger else Logger(self.__class__.__name__)
    self.database = "ned-llm"
    self.initialize_projection()

  def create_mention_pairs(self):
    mentions = [i['id'] for i in self.candidates['entities']]
    mention_pairs = list(itertools.combinations(mentions, 2))

    return mention_pairs

  def create_candidate_pairs(self, pair):
    src_ents = [i['candidates'] for i in self.candidates['entities'] if i['id'] == pair[0]][0]
    src_ids = [i['snomed_id'] for i in src_ents]

    dst_ents = [i['candidates'] for i in self.candidates['entities'] if i['id'] == pair[1]][0]
    dst_ids = [i['snomed_id'] for i in dst_ents]
    return list(itertools.product(src_ids, dst_ids))
  
  def initialize_projection(self):
    query = """
      CALL gds.graph.project(
        'snomedGraph',
        'SnomedEntity',  // Node label
        'SNOMED_RELATION'  // Relationship type
      );
    """
    with self.store._driver.session(database=self.database) as session:
      session.run(query)


  def get_co_occs_query(self, s1_id, s2_id):
    query = f"""
      CALL gds.degree.stream('snomedGraph')
      YIELD nodeId, score
      WITH gds.util.asNode(nodeId).name AS name, score AS degree
      ORDER BY degree DESC
      LIMIT 350
      WITH collect(name) as hub_nodes
      MATCH (s1), (s2)
      WHERE s1.id="{s1_id}" AND
            s2.id="{s2_id}"
      WITH s1, s2, allShortestPaths((s1)-[:SNOMED_RELATION*1..2]-(s2)) AS paths, hub_nodes
      UNWIND paths AS path
      WITH relationships(path) AS path_edges, nodes(path) as path_nodes, hub_nodes
      WITH [n IN path_nodes | n.name] AS node_names,
          [r IN path_edges | r.type] AS rel_types,
          [n IN path_edges | startnode(n).name] AS rel_starts,
          hub_nodes
      WHERE not any(x IN node_names WHERE x IN hub_nodes)
      WITH [i in range(0, size(node_names)-1) | CASE
      WHEN i = size(node_names)-1
      THEN "(" + node_names[size(node_names)-1] + ")"
      WHEN node_names[i] = rel_starts[i]
      THEN "(" + node_names[i] + ")" + '-[:' + rel_types[i] + ']->'
      ELSE "(" + node_names[i] + ")" + '<-[:' + rel_types[i] + ']-' END] as string_paths
      RETURN DISTINCT apoc.text.join(string_paths, '') AS `Extracted paths`
    """.format(s1_id=s1_id, s2_id=s2_id)
    return query

  def get_paths(self):
    with self.store._driver.session(database=self.database) as session:
        paths = []
        mention_pairs = self.create_mention_pairs()

        outer_loop = tqdm(mention_pairs,
                        desc="Processing mention pairs...",
                        position=0,
                        leave=True)
        for i in outer_loop:
            candidate_pairs = self.create_candidate_pairs(i)
            inner_loop = tqdm(candidate_pairs,
                            desc="Processing candidates for each pair...",
                            position=1,
                            leave=False)
            for j in inner_loop:
                query = self.get_co_occs_query(j[0], j[1])
                paths.append(session.run(query))

        cleaned_paths = [sub_item['Extracted paths'] for item in paths for sub_item in item]

        out = []
        for item in cleaned_paths:
            out.append({'id': len(out) + 1, 'path': item})

        return out


class PathTranslation():
    def __init__(self, model, paths, logger=None):
        self.model = model
        self.paths = paths
        self.logger = logger if logger else Logger(self.__class__.__name__)

    def create_paths_to_text_prompt(self):
        system = minify_text("""You are an assistant capable of translating a Neo4j graph path into a clear sentence. 
                                Use the exact entity names from the path while generating the sentence. 
                                The sentences will assist a large language model (LLM) in disambiguating biomedical entities. 
                                Ensure the output is a valid JSON with no extraneous characters.""")

        input = minify_json("""{"path": "(Hypertension)-[:RISK_FACTOR_FOR]->(Cardiovascular Disease)<-[:ASSOCIATED_WITH]-(Myocardial Infarction)"}""")

        assistant = minify_json("""{"sentence": "Hypertension is a risk factor for cardiovascular disease. Myocardial infarction is also associated with cardiovascular disease, indicating that hypertension may increase the risk of experiencing a myocardial infarction through its connection to cardiovascular disease."}""")

        user = minify_json(json.dumps(self.paths))

        return [{"role": "system", "content": system},
                {"role": "user", "content": input},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": user}]

    def translate_paths_to_text(self):
        messages = self.create_paths_to_text_prompt()
        return json.loads(self.model.generate(messages))


class PathSummarization():
    def __init__(self, model, text_paths, logger=None):
        self.model = model
        self.text_paths = text_paths
        self.logger = logger if logger else Logger(self.__class__.__name__)

    def create_summarize_prompt(self):
        system = minify_text("""You are an assistant that can summarize multiple sentences derived from ontology paths into a short summary
                                This summary will be used to support a named entity disambiguation task. 
                                Ensure the output is a valid JSON with no extraneous characters.""")

        input = minify_json("""[
                            {"sentence": "Hypertension is a risk factor for cardiovascular disease. Myocardial infarction is also associated with cardiovascular disease, indicating that hypertension may increase the risk of experiencing a myocardial infarction through its connection to cardiovascular disease."},
                            {"sentence": "Diabetes mellitus is a complication that arises from an endocrine disorder. Diabetic retinopathy is also associated with endocrine disorders, suggesting that diabetes mellitus can lead to the development of diabetic retinopathy through its link to endocrine dysfunction."},
                            {"sentence": "Asthma is associated with respiratory disorders. Allergic rhinitis is also linked to respiratory disorders, which implies that individuals with asthma may also experience allergic rhinitis due to their common association with respiratory conditions."},
                            {"sentence": "Osteoporosis leads to bone weakness. Bone fractures are a result of bone weakness, indicating that osteoporosis can increase the likelihood of bone fractures due to the weakened state of the bones."}
                            ]""")

        assistant = minify_json("""{"context": "Hypertension is a risk factor for cardiovascular disease, which in turn increases the likelihood of experiencing a myocardial infarction. Similarly, diabetes mellitus is linked to endocrine disorders, potentially leading to complications such as diabetic retinopathy. Asthma and allergic rhinitis are both associated with respiratory disorders, suggesting a common link between these conditions. Finally, osteoporosis weakens bones, making individuals more susceptible to bone fractures."}""")

        user = minify_json(json.dumps(self.text_paths))

        return [{"role": "system", "content": system},
                {"role": "user", "content": input},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": user}]

    def summarize_paths(self):
        messages = self.create_summarize_prompt()
        out = self.model.generate(messages)
        return json.loads(out)
