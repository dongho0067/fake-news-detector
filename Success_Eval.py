import pandas as pd
import Detector_AI as dai
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import numpy as np

#This function will take in the results of our ai
#and compare it to the testing data that we have selected. 
def process_results(data, results):
    actual=[]
    predicted=[]

    for x in range(len(data)):
        actual.append(data[x].reals)
        
    for y in range(len(results)):
        predicted.append(results[y].reals)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    cm_display.plot()
    print("The accuracy is:", metrics.accuracy_score(actual, predicted))
    plt.show()
   
