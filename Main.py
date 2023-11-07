#Train the AI on basic input
#Read user input
#parse the input
#Put that input into the AI
#Print out results
import pandas as pd

def main():
    #Read in data set
    #Data was taken from politifacts to determine if it was fake or real
    #Apparently people found it easy to train the AI on the title
    fake_data = pd.read_csv('data/Fake.csv.zip')
    real_data = pd.read_csv('data/True.csv.zip')

    print(fake_data)

if __name__ == "__main__":
    main()