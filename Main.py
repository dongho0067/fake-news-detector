import pandas as pd
import Detector_AI as dai
from newspaper import Article
import Success_Eval
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np

def main():
    print("-0 for article checker")
    print("-1 for testing data")
    mode=input("mode:")

    if(mode=="0"):
        url= input("Enter the url: ")
        article= Article(url, language="en")
        article.download()
        article.parse()
        parse=dai.NewsData(article.title, article.text)
    elif(mode=="1"):
        #dai.training_data()
        #dai.testing_data()
        Success_Eval.process_results(dai.test_data, 0)


if __name__ == "__main__":
    main()