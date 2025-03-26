import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from listing_2 import prompt_segments
#from listing_5 import prompt_segments
#from listing_7 import prompt_segments

_ = load_dotenv()

OPENAI_MODEL = "gpt-4o-mini"

def openai_query(client, prompt_segments: dict, query: str):
    messages = [
        {"role": "system", "content": prompt_segments['task']},
        {"role": "user", "content": prompt_segments['example']},
        {"role": "assistant", "content": prompt_segments['example_output']},
        {"role": "user", "content": query}
    ]
    t_start = time.time()
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0., max_tokens=2000)
    print(response.choices[0].message.content)
    print(f"\nTime: {round(time.time() - t_start, 1)} sec\n")

    return response.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    text = """JOHNS HOPKINS UNIVERSITY Chemistry Department:
Wednesday, November 9, 1932
WW visits the Dept. with Dr. Frazer (Baker Prof. Chem.). There are 239 undergraduate and 116 graduate students of chemistry, the latter group in- cluding holders of the special State fellowships in chemistry under the New
Plan. D.H. Andrews (Prof.Chem.) is a physical chemist specializing in thermodynamics. He is not present at the time of WW's call, but one of his assistants explains his work. He is measuring specific heats of organic compounds by a straight calorimetric method. This work is in its early stages. He is also interested in making mechanical models of various atoms from which can
be demonstrated the theory of the Raman spectra. J.B.Mayer (Assoc. in Chem.) is a former student of G. N. Lewis and works with Max Born at Gottingen summers. He specializes in the energetics of crystal lattices. His wife, last summer, prepared the new edition of Born's treatise on this subject. In Mayer's laboratory Mrs. Wintner, wife of the mathematician, is working on an experimental problem. Andrews says that Mayer is young and impresses one as an enthusiastic and able man."""

    openai_query(client, prompt_segments, text)
    print("-----------\n\n")