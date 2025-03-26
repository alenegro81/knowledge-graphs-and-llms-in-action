from tqdm import tqdm
from logger import Logger
from model import LLM_Model
from util.graphdb_base import GraphDBBase
from ner import NamedEntityRecognition
from ned_cs import CandidateSelection
from path_manager import PathExtraction, PathTranslation, PathSummarization
from ned_dis import CandidateDisambiguation

class NED():
    def __init__(self, model, store, input, logger=None):
        self.model = model
        self.store = store
        self.input = input
        self.logger = logger if logger else Logger(self.__class__.__name__)
    
    def add_candidates(self, sentence, labels):
        for index, value in enumerate(sentence["entities"]):
            candidates = CandidateSelection(self.store).get_candidates(value["mention"], labels, 4)
            sentence["entities"][index]["candidates"] = candidates
    
    def retrieve_paths(self, sentence):
        pe = PathExtraction(self.model, self.store, sentence)
        paths = pe.get_paths()

        return paths
    
    def translate_path(self, path):
        path_translator = PathTranslation(self.model, path)
        text_paths = path_translator.translate_paths_to_text()

        return text_paths
    
    def summarize_paths(self, text_paths):
        summarizer = PathSummarization(self.model, text_paths)
        summary = summarizer.summarize_paths()

        return summary
    
    def disambiguate_entities(self, sentence, context):
        cd = CandidateDisambiguation(self.model, sentence, context)
        disambiguations = cd.disambiguate_paths()

        return disambiguations
    
    def run(self):
        out = []
        
        # Basic splitting - TODO: Could be improved.
        sentences = self.input.split('.')
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        self.logger.info("Number of sentences: " + str(len(sentences)))

        # Named Entity Recognition
        self.logger.info("Executing NER...")
        for sentence in tqdm(sentences):
            ner = NamedEntityRecognition(self.model, self.store, sentence)
            out.append(ner.make_ner())
        
        # Entity labels Extraction
        labels = []
        for i in out:
            for j in i['entities']:
                if j['label'] not in labels:
                    labels.append(j['label'])

        self.logger.info("Extracting candidates for each sentence...")
        
        # Candidate Selection per sentence
        for sentence in out:
            self.add_candidates(sentence, labels)

        self.logger.info("Extracting graph-based context for each sentence...")
        self.logger.info("Number of sentences: " + str(len(out)))
        for sentence in out:
            # Graph paths generation for each sentence
            self.logger.info("  Retrieving paths with Neo4j GDS...")
            paths = self.retrieve_paths(sentence)
            
            # Paths translation into contextual information
            self.logger.info("  Translating paths with LLMs...")
            text_paths = []
            for index, path in enumerate(paths):
                text_path = {
                    "id": index,
                    "sentence": self.translate_path(path)["sentence"]
                }
                text_paths.append(text_path)

            # Summarization of multiple paths
            self.logger.info("  Summarize translated paths with LLMs...")
            summary = self.summarize_paths(text_paths)
            sentence.update(summary)
            for e in sentence["entities"]:
                del e["label"] # Useful to drive the disambiguation

            # Entity Disambiguation
            disambiguations = self.disambiguate_entities(sentence, summary)

            self.logger.info("Disambiguating entities with LLMs...")
            entities = sentence['entities']
            for e in entities:
                for d in disambiguations['entities']:
                    if e['id'] == d['id']:
                        e['disambiguation'] = d['disambiguation']
        
        return out

if __name__ == '__main__':
    model = LLM_Model()
    store = GraphDBBase()
    input = """Zika belongs to the Flaviviridae virus family and it is spread by Aedes mosquitoes.
               Individuals affected by Zika disease and other syndromes like chikungunya fever often experience symptoms like viral myalgia, infectious edema, and infective conjunctivitis.
               Severe outcomes of Zika are due to its capacity to cross the placental barrier during pregnancy, causing microcephaly and congenital malformations."""
    ned = NED(model, store, input)
    res = ned.run()
    print(res)