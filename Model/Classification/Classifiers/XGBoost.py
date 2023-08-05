#https://www.datacamp.com/community/tutorials/xgboost-in-python
#https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7
#https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics
#%%

def XGBoost(X_train, X_test, y_train, y_test):

    model = XGBClassifier()
    model.fit(X_train,y_train)
    
    
    predictions = model.predict(X_test)
    print("XGBoost Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    

    graph_name1= 'XGBoost'+'_without normalization w/o Opt'
    graph_name2=  'XGBoost'
    
    graph_1= 'XGBoost'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= 'XGBoost'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]
    
    for title, normalize ,graphname  in titles_options:
        plt.figure(figsize = (20, 10),dpi=400)
        cm = confusion_matrix(y_test, predictions, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
        disp.plot()

        plt.title(title, size = 12)
        plt.savefig(graphname,bbox_inches='tight',dpi=200)
        plt.show()
        
    savemodel=  'XGBoost'+'_model'+'.sav'    
    joblib.dump(model, savemodel)
    
