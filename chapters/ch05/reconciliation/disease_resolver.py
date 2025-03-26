# noinspection PyUnresolvedReferences
import scispacy
import spacy
# noinspection PyUnresolvedReferences
from scispacy.linking import EntityLinker


class Resolver:
    full = ["Finding", "Organ or Tissue Function", "Tissue"]
    banned = ["Human", "Body Part, Organ, or Organ Component", "Qualitative Concept", "Temporal Concept",
              "Functional Concept", "Body Space or Junction", "Spatial Concept"]

    def __init__(self):
        self.nlp = nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        self.linker = linker = nlp.get_pipe("scispacy_linker")
        self.type_tree = linker.kb.semantic_type_tree
        self.cui_to_entity = self.linker.kb.cui_to_entity

    def canonical(self, entity):
        """get canonical name from entity"""
        if len(entity._.kb_ents) == 0:
            return
        # select the first kb_entity
        return self.cui_to_entity[entity._.kb_ents[0][0]].canonical_name

    @staticmethod
    def hasEntities(item):
        """return true if item has at least one recognised entity"""
        return len(item.ents) > 0

    def types(self, entity):
        """return semantic types for the entity"""
        if len(entity._.kb_ents) == 0:
            return []
        return [self.type_tree.get_canonical_name(t)
                for t in self.cui_to_entity[entity._.kb_ents[0][0]].types]

    @staticmethod
    def matchesAll(entity):
        """return trie if the entity covers the whole content"""
        return entity.start == 0 and entity.end == len(entity.doc)

    def containsOnly(self, entity, targets):
        """return true if the entity types are within the target types """
        return set(self.types(entity)).intersection(targets) == set(self.types(entity))

    def validEntity(self, entity):
        """ exploits the entity types to detect if an entity is corretly identified as disease """
        # "banned" types can not be related with diseases
        if self.containsOnly(entity, self.banned):
            return False
        # "full" types can be considered valid if they matche exactly the whole text
        if self.containsOnly(entity, self.full):
            return self.matchesAll(entity)
        return True

    def normalize(self, item):
        """"main entrypoint: convert item into a normalized disease
            return ( normalized_name, UMNLS_ID | None )
        """
        # single entity found
        if len(item.ents) == 1:
            return self.normalize_entity(item)
        # multiple entityies found
        if len(item.ents) > 1:
            return self.normalize_default(item)
        # otherwise
        return self.normalize_default(item)

    def normalize_entity(self, item):
        """ normalize item when there is only one detected entity """
        entity = item.ents[0]
        if self.validEntity(entity):
            return self.canonical(entity), entity._.kb_ents[0][0]
        # invalid entity switch to default
        return self.normalize_default(item)

    def normalize_default(self, item):
        """ when no other better options are available return capitalized version """
        item = str(item)
        item = " ".join(i.capitalize() for i in item.split())
        return item.strip(), None
