# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 02:51:37 2023

@author: srpv
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics


def RF(X_train, X_test, y_train, y_test,n_estimators,feature_cols):
    
    model = RandomForestClassifier(n_estimators=n_estimators , oob_score=True)
    model.fit(X_train, y_train)
            
    #Accuracy of the model
    
    predictions = model.predict(X_test)
    print("RF Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    
    #Plotting of the model
    
    graph_name1= 'Random Forest'+'_without normalization w/o Opt'
    graph_name2=  'Random Forest'
    
    graph_1= 'Random Forest'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= 'Random Forest'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=200)
        
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
        disp.plot()

        plt.title(title, size = 12)
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        plt.show()
    
    
    savemodel=  'Random Forest'+'_model'+'.sav'
    joblib.dump(model, savemodel)


    
