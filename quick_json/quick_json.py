__all__ = ["json_encode"]

import json as json

def json_encode(obj, filename, indent=0):
    JsonStr = json.dumps(obj, separators=(',', ':'), sort_keys=True, indent=2)
    with open(filename, "w") as TheFile:
        TheFile.write(JsonStr)

def test():
    json_encode([2, 3, 4, {"that": "boo"}], "experiment.json")
    pass