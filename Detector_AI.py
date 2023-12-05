import pandas as pd
import torch

fake_data=[]
real_data=[]
mixed_data=[]

#Create classes 
class NewsData:
    def __init__(self, title:str, text:str, subject:str, date:str):
        self.title=title
        self.text=text
        self.subject=subject
        self.date=date


def training_data():
    #Read in data set to train AI on
    fd = pd.read_csv('data/Fake.csv.zip')
    rd = pd.read_csv('data/True.csv.zip')

    #Need to create a data set to combines both
    #Parse the data and convert it into a dictionary to use
    #Now you can iterate through the keys of the dict for your AI
    fdata=fd.to_dict()
    rdata=rd.to_dict()

    
    fake_data=[]
    real_data=[]
    mixed_data=[]

    #Create and store all the data to make classes
    titles=[]
    texts=[]
    subjects=[]
    dates=[]
    
    #What do we do with rdata? fdata is actual data to use
    for data in fdata, rdata:
        for keys in data:
            for values in fdata.get(keys):
                if(keys=="title"):
                    titles.append(data.get(keys).get(values))
                elif(keys=="text"):
                    texts.append(data.get(keys).get(values))
                elif(keys=="subject"):
                    subjects.append(data.get(keys).get(values))
                else:
                    dates.append(data.get(keys).get(values))

    for x in range(len(titles)):
        if x<len(fdata.get("title").keys()):        
            fake_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
        else:
            real_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
        if x%2==0:
            mixed_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
