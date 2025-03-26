import json
import re

def minify_json(json_str):
    parsed = json.loads(json_str)
    minified_json = json.dumps(parsed, separators=(',', ':'))
    return minified_json

def minify_text(text_str):
    minified_text = re.sub(r'\s+', ' ', text_str)
    return minified_text