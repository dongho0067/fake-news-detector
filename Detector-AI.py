import pandas as pd

fake_data=[]
real_data=[]
mixed_data=[]

#Create classes 
class NewsData:
    def __init__(title:str, text:str, subject:str, date:str):
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

    #Create and store all the data to make classes
    titles=[]
    texts=[]
    subjects=[]
    dates=[]

    for data in fdata, rdata:
        for keys in data:
            for values in fdata.get(keys):
                if(keys=="title"):
                    titles.append(fdata.get(keys).get(values))
                elif(keys=="text"):
                    texts.append(fdata.get(keys).get(values))
                elif(keys=="subjects"):
                    subjects.append(fdata.get(keys).get(values))
                else:
                    subjects.append(fdata.get(keys).get(values))

    #Put the data into their objects
    for x in range(len(titles)):
        if x<len(fdata.get("title").keys()):
            fake_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
        else:
            real_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
        if x%2==0:
            mixed_data.append(NewsData(titles[x], texts[x], subjects[x], dates[x]))
        

def detector(data:dict):
    #This is where the AI function will go
    for keys in data:
        pass



#Results:
#The final results of the AI should output:
#The data set used, the number of fake articles found
