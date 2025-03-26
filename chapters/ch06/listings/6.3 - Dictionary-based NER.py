import spacy
from spacy import displacy

if __name__ == "__main__":
    text = "Jane Austen, the Victorian era writer, works nowadays for Google."

    # load standard English NLP and NER models
    # start with downloading the models by running this command: `python -m spacy download en_core_web_md`
    # nlp = spacy.load("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    ruler = nlp.add_pipe("entity_ruler")
    patterns = [
        {"pattern": [{"LEMMA": "writer"}], "label": "TITLE"}
    ]
    ruler.add_patterns(patterns)

    doc = nlp(text)

    print("\n".join([f"{en.text}\t{en.label_}\t-\t{spacy.explain(en.label_)}" for en in doc.ents]))

    # visualise named entities
    displacy.serve(doc, style="ent", auto_select_port=True)