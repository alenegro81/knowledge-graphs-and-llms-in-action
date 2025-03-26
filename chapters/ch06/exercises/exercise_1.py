import spacy
from spacy import displacy


def regex_ner(text):
    # Solution of the exercise in chapter 6.1.2

    # load standard English NLP and NER models
    # download the model by running this command: `python -m spacy download en_core_web_sm`
    nlp = spacy.load("en_core_web_sm")

    # add entity ruler to the pipeline
    ruler = nlp.add_pipe("entity_ruler")

    # define NER dictionary patterns for C-level business titles
    patterns = [{"pattern": [{"LEMMA": "writer"}], "label": "TITLE"}]
    patterns.append({"pattern": [{"LOWER": "chief"}, {"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}, {"LOWER": "officer"}], "label": "TITLE"})
    patterns.append({"pattern": [{"LOWER": "chief"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}, {"LOWER": "officer"}], "label": "TITLE"})
    patterns.append({"pattern": [{"TEXT": {"REGEX": "C[A-Z]O"}}], "label": "TITLE"})
    ruler.add_patterns(patterns)

    doc = nlp(text)

    print("\n".join([f"{en.text}\t{en.label_}\t-\t{spacy.explain(en.label_)}" for en in doc.ents]))
    #displacy.serve(doc, style="dep") #, port=1111) # for visual representation of the dependency parser output


if __name__ == "__main__":
    ### Solution of the exercise in chapter 6.1.2
    text = "Jane Austen, the Victorian era writer and Chief data officer and CEO, works nowadays for Google."
    regex_ner(text)