import spacy


class EntityExtractor:
    """Convert a text into a structured dictionary of entities using scispacy and the
       scispacy linker to UMLS concepts"""
    def __init__(self, metathesaurus_file):
        self.types = {}
        print('scispacy model loading...')
        self.nlp = spacy.load("en_core_sci_md")
        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = self.nlp.get_pipe("scispacy_linker")
        self.linker_dict = self.linker.kb.cui_to_entity      
        self.process_metathesaurus(metathesaurus_file)

    def process_metathesaurus(self, metathesaurus_file):
        """Process the UMLS meta thesaurus
           Create a mapping from CuIDs returned by scispacy linker to a descriptive string"""
        for line in metathesaurus_file.open():
            values = line.split('|')
            self.types[values[2]] = values[3].strip()

    def create_entity_dict(self, doc, ent):
        """collect the data from entity object from scispacy into structured dictionary"""
        if len(ent._.kb_ents) == 0:
            return

        ent_dict = {
            'sentenceIndex': [s.text for s in doc.sents].index(ent.sent.text),
            'value': " ".join(ent.text.split()).strip(),
            'lemma': " ".join(ent.lemma_.split()).strip(),
            'label': ent.label_.upper(),
            'beginCharacter': ent.start_char,
            'endCharacter': ent.end_char,
            'ned': []
        }
        for umls_ent in ent._.kb_ents:
            target = self.linker.kb.cui_to_entity[umls_ent[0]]
            ent_dict['ned'].append({
                'id': target.concept_id,
                'name': target.canonical_name.strip(),
                'definition': target.definition,
                'aliases': target.aliases,
                'types': target.types,
                'confidence': umls_ent[1]
            })

        if ent_dict['ned'][0]['confidence'] < 0.8:
            return

        ent_dict['selected_ned_id'] = ent_dict['ned'][0]['id']
        ent_dict['selected_ned_name'] = ent_dict['ned'][0]['name']
        ent_dict['selected_ned_definition'] = ent_dict['ned'][0]['definition']
        ent_dict['selected_ned_aliases'] = ent_dict['ned'][0]['aliases']
        ent_dict['selected_ned_types_id'] = ent_dict['ned'][0]['types']
        ent_dict['selected_ned_types'] = [self.types[i] for i in ent_dict['ned'][0]['types']]
        ent_dict['selected_ned_type'] = self.types[ent_dict['ned'][0]['types'][0]]

        return ent_dict

    def extract_ents(self, text):
        """convert text into a structured dictionary of entities"""
        doc = self.nlp(text)
        ents = []
        for e in doc.ents:
            ent = self.create_entity_dict(doc, e)
            if ent is not None:
                ents.append(ent)

        return ents
