import sys
from datetime import datetime

import yaml
from yaml import SafeLoader

try: #LOAD CONFIG FILE
    with open(sys.argv[1],encoding="utf-8") as f:
        print(f"Loading config file {sys.argv[1]}")
        data_yml = yaml.load(f, Loader=SafeLoader)
        print("Config file loaded successfully")
        print(f"Archive: {data_yml['data']}")
except: #IF NO FILE IS PASSED AS ARGUMENT
    print("Error loading config file")
    exit()
def combine_dic(dic1,dic2,dic3={},dic4={}): #COMBINE DICTIONARIES
    dic = {**dic1, **dic2, **dic3, **dic4}
    return dic

def diff_bewteen_dates(date1,date2): #DIFF BETWEEN DATES
    date1 = datetime.strptime(date1, '%d/%m/%Y')
    diff = date2 - date1
    return diff.days

def make_date_from_objetive(): #MAKE DATE FROM OBJETIVE
    objetive= (datetime.strptime(data_yml["objetive_date"], '%d/%m/%Y'))
    return objetive