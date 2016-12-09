#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Genre Classifier

Watch README.md for more informations

@author: Daniele Gamba
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from FeaturesExtraction import extractFeatures

nFeatures = 9
storage = "model.p"

class Error(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def main(command=None):
    args = None
    if not command:
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", help="Train, test and classify", nargs='+', choices=["train","predict"])
        #parser.add_argument("filename", help="File to be labeled by the model",nargs='?', default="none")
        args = parser.parse_args()
        command = args.mode[0]
    
    dct = {"train":train,"predict":logRegPredict}
    return dct[command]()

    
def createDataset():
    print("---------------------------------------------------")
    print("Place your labeled .wav music in the .data/ folder")
    print("to convert your music from mp3 to wav use:")
    print("sox input.mp3 output.wav channels 1")
    print("---------------------------------------------------")
    
    files = sorted(os.listdir("data"))
    files = [file for file in files if file.endswith(".wav")]
    
    if nFeatures > len(files):
        raise Error("Too many features", "Change you desidered number of features or add some file")         
    dataset = []
    labels = {}
    nlabels = 0
    for indx, file in enumerate(files):
        print(str(indx)+" "+file)

        catname = file.split("_")[0]
        
        if catname in labels:
            catvalue = labels[catname]
        else:
            nlabels+=1
            labels[catname] = nlabels*10
            catvalue = labels[catname]
            
        dataset.append([catvalue, extractFeatures(str("data/"+file), nFeatures)])
        #plt.plot(dataset[indx][1], label=file)
    #plt.legend()
    return dataset, labels
    
    
def logRegression(x, y, labels):
    logreg = linear_model.LogisticRegression()
    logreg.fit(x,y)  #X, Y
    print("Prediction score (on training set): "+str(logreg.score(x,y)))
    with open(storage, "wb") as f:
        pickle.dump(logreg, f)
        pickle.dump(labels, f)

    
def logRegPredict(logreg=None):
    with open(storage, "rb") as f:
        logreg = pickle.load(f)
        labels = pickle.load(f)
    files = os.listdir("predict")
    files = [file for file in files if file.endswith(".wav")]
    print("--- Prediction test ---")
    print("Classes are")
    for label in labels:
        print("["+str(labels[label])+"] "+label)
    print("")
        
    for file in files:
        x = extractFeatures("predict/"+file, nFeatures)
        x = np.matrix(x)  
        result = logreg.predict(x)
        result_prob = logreg.predict_proba(x)
        #print("- "+file+": "+str(result)+" - with prob. "+str(result_prob))    #print all the crossed probability
        print("- "+file+": "+str(result)+" - with prob. [%] " + str(100*result_prob[0,(int(result/10) - 1)]))
        
    
def train():
    dataset, labels = createDataset()
    nEntry = len(dataset)
    
    #logisticRegression
    x = []
    y = []
    for i in range(nEntry):
        x.append(dataset[i][1])
        y.append(dataset[i][0])
    x = np.matrix(x)
    y = np.ravel(y)
    logRegression(x,y, labels)

    
if __name__ == '__main__':
    #dataset = main("train") #use this line to use it inside an editor (like Spyder)
    main()