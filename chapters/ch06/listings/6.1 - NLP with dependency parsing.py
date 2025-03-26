import spacy
from spacy import displacy
# https://spacy.io/usage/visualizers


if __name__ == "__main__":
    text = "Jane Austen, the Victorian era writer, works nowadays for Google."

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print(doc)

    displacy.serve(doc, style="dep")  # options={'compact': True})