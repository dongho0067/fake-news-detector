import pandas as pd
import torch
import math

fake_data=[]
real_data=[]

test_data=[]

#Create class to parse our data. real:0=Fake, 1=Real
class NewsData:
    def __init__(self, title:str, text:str, real:int):
        self.title=title
        self.text=text
        self.real=real


def training_data():
    #Read in data set to train AI on
    fd = pd.read_csv('data/Fake.csv.zip')
    rd = pd.read_csv('data/True.csv.zip')

    #Need to create a data set to combines both
    #Parse the data and convert it into a dictionary to use
    #Now you can iterate through the keys of the dict for your AI
    fdata=fd.to_dict()
    rdata=rd.to_dict()

    #Create and store all the data to make classes
    titles=[]
    texts=[]

    
    #What do we do with rdata? fdata is actual data to use
    for data in fdata, rdata:
        for keys in data:
            for values in fdata.get(keys):
                if(keys=="title"):
                    titles.append(data.get(keys).get(values))
                elif(keys=="text"):
                    texts.append(data.get(keys).get(values))


    for x in range(len(titles)):
        if x<len(fdata.get("title").keys()):        
            fake_data.append(NewsData(titles[x], texts[x], 0))
        else:
            real_data.append(NewsData(titles[x], texts[x], 1))

def testing_data():
    #Read in data set to train AI on
    td = pd.read_csv('data/Test.csv.zip')
    
    #1==Fake News!
    #Need to create a data set to combines both
    #Parse the data and convert it into a dictionary to use
    #Now you can iterate through the keys of the dict for your AI
    tdata=td.to_dict()

    #Create and store all the data to make classes
    titles=[]
    texts=[]
    reals=[]
    
    for keys in tdata:
        for values in tdata.get(keys):
            if(keys=="title"):
                titles.append(tdata.get(keys).get(values))
            elif(keys=="text"):
                texts.append(tdata.get(keys).get(values))
            elif(keys=="label"):
                #The data table accidentally reversed 0 and 1 for Real/Fake
                if tdata.get(keys).get(values)==0:
                    reals.append(1)
                else:
                    reals.append(0)

    for x in range(len(titles)):  
        #Some test data is just missing titles/texts, so we will make sure 
        #to only include data that has both the title and text
        if(isinstance(titles[x], str) and isinstance(texts[x], str)):
            test_data.append(NewsData(titles[x], texts[x], reals[x]))

