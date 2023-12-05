#Train the AI on basic input
#Read user input
#parse the input
#Put that input into the AI
#Print out results
import pandas as pd
import Detector_AI as dai
from newspaper import Article
from Success_Eval import process_results

def main():
    url= input("Enter the url: ")
    date= input("Enter the date(Ex. December 17th, 2020): ")
    article= Article(url, language="en")
    article.download()
    article.parse()
    parse=dai.NewsData(article.title, article.text, "News", date)



if __name__ == "__main__":
    main()