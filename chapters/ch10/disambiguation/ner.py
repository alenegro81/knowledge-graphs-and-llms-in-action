import json
from utils import minify_json, minify_text
from logger import Logger

class NamedEntityRecognition:
    def __init__(self, model, store, sentence, logger=None):
        self.database = "ned-llm"
        self.model = model
        self.store = store
        self.sentence = sentence
        self.logger = logger if logger else Logger(self.__class__.__name__)

    def named_entity_query(self):
        query = """
        MATCH (n:SnomedEntity)
        UNWIND n.type as named_entity
        WITH DISTINCT named_entity, count(named_entity) as num_of_entities
        ORDER BY num_of_entities DESC
        RETURN collect(named_entity) as named_entities
        """

        return query

    def retrieve_named_entities(self):
        self.logger.info("Retrieving named entities from Neo4j...")
        query = self.named_entity_query()
        with self.store._driver.session(database=self.database) as session:
            return session.run(query).single()["named_entities"]

    def find_all_mention_indices(self, string, substring):
        indices = []
        start_index = 0

        while True:
            start_index = string.find(substring, start_index)

            if start_index == -1:
                break  # No more occurrences found

            end_index = start_index + len(substring) - 1
            indices.append((start_index, end_index))

            # Move start_index forward to search for the next occurrence
            start_index += len(substring)

        return indices

    def create_ner_prompt(self):
        named_entities = self.retrieve_named_entities()

        system = minify_text("""You are an AI assistant capable of extracting named entities in the medical domain. 
                                Your task is to extract ALL single mentions of named entities from the text. 
                                You must only use one of the pre-defined entities from the following list: {named_entities}. No other entity categories are allowed.
                                For each sentence, extract the named entities and present the output in valid JSON format""".format(named_entities=named_entities))

        input = minify_text("""Risk factors for rhinocerebral mucormycosis include poorly controlled diabetes mellitus and severe immunosuppression.
                            The administration of corticosteroids for severe COVID-19 treatment can disrupt blood glucose control in diabetes patients,
                            leading to an increased risk of rhinocerebral mucormycosis.
                            Additionally, Rhizopus oryzae, one of the most common fungi responsible for mucormycosis, thrives in environments with elevated exposure to organic matter,
                            which is often encountered in agricultural settings.
                            Contributing factors to the outbreak of rhinocerebral mucormycosis in India include the high prevalence of diabetes mellitus, elevated environmental exposure,
                            and the widespread use of corticosteroids during the COVID-19 epidemic.
                            Additionally, the frequent activity of agricultural work, the common behavior of not wearing masks,
                            and the governmental regulations on steroid use have exacerbated the situation, along with the anatomical vulnerability of nasal cavities and sinuses to fungal infections.""")

        assistant = minify_json("""[{"sentence": "Risk factors for rhinocerebral mucormycosis include poorly controlled diabetes mellitus and severe immunosuppression.", "entities": [{"id": 0, "mention": "Risk factors", "label": "Events"}, {"id": 1, "mention": "rhinocerebral mucormycosis", "label": "Disease"}, {"id": 2, "mention": "poorly controlled diabetes mellitus", "label": "Disease"}, {"id": 3, "mention": "severe immunosuppression", "label": "Qualifier value"}]}, {"sentence": "The administration of corticosteroids for severe COVID-19 treatment can disrupt blood glucose control in diabetes patients, leading to an increased risk of rhinocerebral mucormycosis.", "entities": [{"id": 4, "mention": "administration", "label": "Procedure"}, {"id": 5, "mention": "corticosteroids", "label": "Substance"}, {"id": 6, "mention": "COVID-19", "label": "Disease"}, {"id": 7, "mention": "treatment", "label": "Procedure"}, {"id": 8, "mention": "blood glucose control", "label": "Events"}, {"id": 9, "mention": "diabetes", "label": "Disease"}, {"id": 10, "mention": "rhinocerebral mucormycosis", "label": "Disease"}]}, {"sentence": "Additionally, Rhizopus oryzae, one of the most common fungi responsible for mucormycosis, thrives in environments with elevated exposure to organic matter, which is often encountered in agricultural settings.", "entities": [{"id": 11, "mention": "Rhizopus oryzae", "label": "Organism"}, {"id": 12, "mention": "fungi", "label": "Organism"}, {"id": 13, "mention": "mucormycosis", "label": "Disease"}, {"id": 14, "mention": "environments", "label": "Environments and geographical locations"}, {"id": 15, "mention": "elevated exposure", "label": "Events"}, {"id": 16, "mention": "organic matter", "label": "Substance"}, {"id": 17, "mention": "agricultural settings", "label": "Environments and geographical locations"}]}, {"sentence": "Contributing factors to the outbreak of rhinocerebral mucormycosis in India include the high prevalence of diabetes mellitus, elevated environmental exposure, and the widespread use of corticosteroids during the COVID-19 epidemic.", "entities": [{"id": 18, "mention": "Contributing factors", "label": "Events"}, {"id": 19, "mention": "outbreak", "label": "Events"}, {"id": 20, "mention": "rhinocerebral mucormycosis", "label": "Disease"}, {"id": 21, "mention": "diabetes mellitus", "label": "Disease"}, {"id": 22, "mention": "elevated environmental exposure", "label": "Events"}, {"id": 23, "mention": "widespread use", "label": "Events"}, {"id": 24, "mention": "corticosteroids", "label": "Substance"}, {"id": 25, "mention": "COVID-19 epidemic", "label": "Events"}]}, {"sentence": "Additionally, the frequent activity of agricultural work, the common behavior of not wearing masks, and the governmental regulations on steroid use have exacerbated the situation, along with the anatomical vulnerability of nasal cavities and sinuses to fungal infections.", "entities": [{"id": 26, "mention": "frequent activity", "label": "Events"}, {"id": 27, "mention": "agricultural work", "label": "Events"}, {"id": 28, "mention": "governmental regulations", "label": "Events"}, {"id": 29, "mention": "steroid use", "label": "Events"}, {"id": 30, "mention": "situation", "label": "Events"}, {"id": 31, "mention": "anatomical vulnerability", "label": "Events"}, {"id": 32, "mention": "nasal cavities", "label": "Body structure"}, {"id": 33, "mention": "sinuses", "label": "Body structure"}, {"id": 34, "mention": "fungal infections", "label": "Disease"}]}]""")

        user = minify_text(self.sentence)

        return [{"role": "system", "content": system},
                {"role": "user", "content": input},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": user}]

    def make_ner(self):
        # TODO improve to manage identical mentions multiple times
        messages = self.create_ner_prompt()
        self.logger.info("Performing NER with the LLM...")
        out = json.loads(self.model.generate(messages))[0]
        sentence = out['sentence']
        entities = out['entities']
        for index, ent in enumerate(entities):
            chars_index = self.find_all_mention_indices(sentence, ent['mention'])[0]
            entities[index]['start'] = chars_index[0]
            entities[index]['end'] = chars_index[1]
        
        return out