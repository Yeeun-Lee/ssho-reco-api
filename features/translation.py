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
    items = pd.DataFrame(items)
    return clean(items)

def cleanColors(text):
    return re.sub(" /.+$| \\[.+\\]| [A-Z]+? \\[.+\\]|^[A-Za-z]+[0-9]+? | -.+$", "", text)

def finalItems(file = None):
    date = str(datetime.now().strftime("%y%m%d"))
    if file==None:
        items = getItems()
        items.to_csv("../assets/processed_{}.csv".format(date), index = False)
    else:
        items = pd.read_csv(file)

    items['_title'] = items['title'].apply(lambda x: cleanColors(x))
    new_title = []
    count = 0
    for _id, title in items[['id', '_title']].values:
        count+=1
        print("==== itemId : {}, count : {}".format(_id, count))
        print("Original title : ", title)
        new = toEng(title)
        print("Translated : ", new)
        print()
        new_title.append(new)
    items['translated'] = new_title
    items.to_csv("../assets/translation_{}.csv".format(date), index = False)
    print("DONE")

if __name__=="__main__":
    finalItems()