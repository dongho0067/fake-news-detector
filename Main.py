#Train the AI on basic input
#Read user input
#parse the input
#Put that input into the AI
#Print out results
import pandas as pd
import Detector_AI as dai
from newspaper import Article
import Success_Eval

def main():
    mode=input("Enter 0 for article or enter 1 for data testing: ")

    if(mode=="0"):
        url= input("Enter the url: ")
        article= Article(url, language="en")
        article.download()
        article.parse()
        parse=dai.NewsData(article.title, article.text)
    elif(mode=="1"):
        dai.training_data()
        dai.testing_data()
        #Success_Eval.process_results(dai.test_data)


if __name__ == "__main__":
    main()