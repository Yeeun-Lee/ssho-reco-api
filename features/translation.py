import pandas as pd
import requests
import json
import time
from datetime import datetime

import re
from google_trans_new import google_translator

URL = "http://api.ssho.tech:8080/item"

def toEng(title):
    translator = google_translator()
    return translator.translate(title, lang_tgt="en")

def clean(my_dict):
    try:
        del my_dict['productExtra']
        del my_dict['imageUrl']
        del my_dict['link']

    except KeyError:
        pass
    return my_dict

def getItems():
    response = requests.get(URL)
    items = response.json()
    items = [clean(x) for x in items]
    return items

def cleanColors(text):
    return re.sub(" /.+$| \\[.+\\]| [A-Z]+? \\[.+\\]|^[A-Za-z]+[0-9]+? | -.+$", "", text)

def finalItems(file = None):
    date = str(datetime.now().strftime("%y%m%d"))
    if file==None:
        items = getItems()
        with open("processed_{}.json".format(date), "w") as f:
            json.dump(items, f)
    else:
        with open(file, "r") as f:
            items = json.load(f)

    for i in range(len(items)):
        items[i]['_title'] = cleanColors(items[i]['title'])
        print("==== itemId : {}, count : {}".format(items[i]['id'], i+1))
        print("Original title : ", items[i]['_title'])
        new = toEng(items[i]['_title'])
        print("Translated : ", new)
        items[i]['translated'] = new
        print()

    with open("translated_{}.json".format(date), "w") as f:
        json.dump(items, f)

    return items

if __name__=="__main__":
    finalItems()