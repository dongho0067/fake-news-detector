#Train the AI on basic input
#Read user input
#parse the input
#Put that input into the AI
#Print out results
import pandas as pd
from Success_Eval import process_results

def main():
    fd = pd.read_csv('data/Fake.csv.zip')
    rd = pd.read_csv('data/True.csv.zip')

    #Need to create a data set to combines both

    #Parse the data and convert it into a dictionary to use
    #Now you can iterate through the keys of the dict for your AI
    fdata=fd.to_dict()
    rdata=rd.to_dict()

    print(len(fdata.get("title").keys()))



if __name__ == "__main__":
    main()