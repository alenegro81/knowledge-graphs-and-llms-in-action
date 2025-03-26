import requests
import json

url = "https://api.ubiai.tools:8443/api_v1/annotate/"
my_token = "XXX" # enter your UBIAI API token


if __name__ == "__main__":
    data = {
        "inputs" : [ # enter texts you want to process
            "<put your text here>",
            "<put more text here>"
        ]
    }

    response = requests.post(url + my_token,json=data) #C

    res = json.loads(response.content.decode("utf-8"))