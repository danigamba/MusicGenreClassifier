#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Genre Classifier

Watch README.md for more informations

@author: Daniele Gamba
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

from FeaturesExtraction import extractFeatures

nFeatures = 9

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
    
    dct = {"train":train,}
    return dct[command]()

    
def createDataset():
    print("---------------------------------------------------")
    print("Place your labeled .wav music in the .data/ folder")
    print("to convert your music from mp3 to wav use:")
    print("sox input.mp3 output.wav channels 1")
    print("---------------------------------------------------")
    
    files = os.listdir("data")
    files = [file for file in files if file.endswith(".wav")]
    
    if nFeatures > len(files):
        raise Error("Too many features", "Change you desidered number of features or add some file")         
    dataset = []
    labels = []
    nlabels = 0
    for indx, file in enumerate(files):
        print(str(indx)+" "+file)

        catname = file.split("_")[0]
        if catname == "Classical":
            catvalue = 0
        elif catname == "Metal":
            catvalue = 100
        else:
            catvalue = 50
        
        if catname not in labels:
            nlabels+=1
            labels.append(catname)
            
        dataset.append([catvalue, extractFeatures(str("data/"+file), nFeatures)])
        #plt.plot(dataset[indx][1], label=file)
        
    #plt.legend()
    return dataset, nlabels
    
    
def nn_model(nFeatures):
    input_layer = tflearn.input_data(shape=[None,nFeatures], name='input')
    fullyconn1 = tflearn.fully_connected(input_layer, 1, name='fullyconn')
    #fullyconn2 = tflearn.fully_connected(fullyconn1, 1, name='fullyconn')
    regression = tflearn.regression(fullyconn1, optimizer='adam',
                                learning_rate=0.01,
                                loss='categorical_crossentropy')
    model = tflearn.DNN(regression, checkpoint_path='model.tfl.ckpt')
    return model
    

def logRegression(x, y):
    logreg = linear_model.LogisticRegression()
    logreg.fit(x,y)  #X, Y
    print("Prediction score (on training set): "+str(logreg.score(x,y)))
    #print(logreg.get_params())
    logRegPredict(logreg)

    
def logRegPredict(logreg):
    files = os.listdir("predict")
    files = [file for file in files if file.endswith(".wav")]
    print("--- Prediction test ---")
    print("Classes are\n[0] Classical\n[50] Other\n[100] Metal")
    for file in files:
        x = extractFeatures("predict/"+file, nFeatures)
        x = np.matrix(x)  
        result = logreg.predict(x)
        result_prob = logreg.predict_proba(x)
        print("- "+file+": "+str(result)+" - with prob. "+str(result_prob))
    
    
def train():
    dataset, nlabels = createDataset()
    nEntry = len(dataset)
    #TODO: add the unknown labels to features for the entry with catvalye == 50
    
    #logisticRegression
    x = []
    y = []
    for i in range(nEntry):
        x.append(dataset[i][1])
        y.append(dataset[i][0])
    x = np.matrix(x)
    y = np.ravel(y)
    logRegression(x,y)
        
    
if __name__ == '__main__':
    #dataset = main("train") #use this line to use it inside an editor (like Spyder)
    main()