import json
from utils import minify_json, minify_text
from logger import Logger

class CandidateDisambiguation():
  def __init__(self, model, sentence, context, logger=None):
    self.model = model
    self.sentence = sentence
    self.context = context
    self.logger = logger if logger else Logger(self.__class__.__name__)

  def create_disambiguation_prompt(self):
    system = minify_text("""You are an assistant specialized in entity disambiguation. Your task is to identify and accurately disambiguate the entities mentioned in a given sentence, relying heavily on the contextual entities present in surrounding sentences:
                         1. Original Sentence: The sentence that contains ambiguous entities that need to be resolved.
                         2. Candidate Entities: A list of potential entities extracted from the sentence, with each entity having multiple possible meanings or labels.
                         3. Contextual Sentences: A collection of related or surrounding sentences that provide additional context for disambiguating the mentioned entities.
                         Your objective is to use the entities mentioned in the contextual sentences as the primary source of information to disambiguate the entities in the original sentence. Analyze the candidate entities for each ambiguous mention and select the one that aligns best with both the context and the meaning provided by the contextual sentences.
                         The output must be a valid JSON.""")
    
    input = minify_json("""{"sentence":"Asthma and allergic rhinitis are commonly addressed together in treatment protocols, given their shared underlying inflammatory processes in allergic individuals.","candidates":[{"id":1,"candidates":[{"snomed_id":"233681001","name":"Extrinsic asthma with asthma attack"},{"snomed_id":"195967001","name":"Asthma"},{"snomed_id":"266361008","name":"Intrinsic asthma"},{"snomed_id":"266364000","name":"Asthma attack"},{"snomed_id":"270442000","name":"Asthma monitored"},{"snomed_id":"170642006","name":"Asthma severity"},{"snomed_id":"170643001","name":"Occasional asthma"},{"snomed_id":"170644007","name":"Mild asthma"},{"snomed_id":"170645008","name":"Moderate asthma"}]},{"id":2,"candidates":[{"snomed_id":"196219007","name":"[X]Other allergic rhinitis"},{"snomed_id":"311000119101","name":"Non-allergic rhinitis"},{"snomed_id":"446096008","name":"Perennial allergic rhinitis (disorder)"},{"snomed_id":"61582004","name":"Allergic rhinitis"},{"snomed_id":"703903005","name":"Dermatophagoides farinae allergic rhinitis"},{"snomed_id":"196218004","name":"[X]Other seasonal allergic rhinitis"},{"snomed_id":"10629391000119103","name":"Allergic rhinitis caused by feathers"},{"snomed_id":"195821000","name":"Allergic rhinitis NOS"},{"snomed_id":"21719001","name":"Allergic rhinitis due to pollen"}]}],"context":"Asthma is associated with respiratory disorders. Allergic rhinitis is also linked to respiratory disorders, which implies that individuals with asthma may also experience allergic rhinitis due to their common association with respiratory conditions."}""")

    assistant = minify_json("""{"entities":[{"id":1,"disambiguation":{"snomed_id":"195967001","name":"Asthma"}},{"id":2,"disambiguation":{"snomed_id":"61582004","name":"Allergic rhinitis"}}]}""")

    user = minify_json(json.dumps(self.sentence))

    return [{"role": "system", "content": system},
            {"role": "user", "content": input},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": user}]

  def disambiguate_paths(self):
    messages = self.create_disambiguation_prompt()
    out = self.model.generate(messages)
    return json.loads(out)
