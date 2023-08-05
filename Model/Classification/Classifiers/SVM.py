#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
from sklearn import metrics

def SVM(X_train, X_test, y_train, y_test,Featurespace, classspace):
    random_state = np.random.RandomState(0)
    #svc_model = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale',decision_function_shape='ovo', verbose=True,random_state=None)
    svc_model = SVC(kernel='rbf', probability=True, random_state=random_state)
    model=svc_model.fit(X_train,y_train)
    
    predictions = model.predict(X_test)
    print("SVM Accuracy:",metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))
    
    graph_name1= 'SVM'+'_without normalization w/o Opt'
    graph_name2=  'SVM'
    
    graph_1= 'SVM'+'_Confusion_Matrix'+'_'+'No_Opt'+'.png'
    graph_2= 'SVM'+'_Confusion_Matrix'+'_'+'Opt'+'.png'
    
    
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
        
    savemodel=  'SVM'+'_model'+'.sav'
    joblib.dump(model, savemodel)
    

