# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
np.random.seed(1974)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def TSNEplot(X_train,y_train,graph_name,graph_title,perplexity):
    

    print('output shape: ', X_train.shape)
    print('target shape: ', y_train.shape)
    print('perplexity: ',perplexity)
    

    y_train = np.ravel(y_train)
    

    RS=np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(X_train)
    np.save('tsne_3d.npy',tsne_fit)
    tsne_fit = np.load('tsne_3d.npy')
    
    x1=tsne_fit[:, 0]
    x2=tsne_fit[:, 1]
    x3=tsne_fit[:, 2]
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=y_train))
    groups = df.groupby('label')
    
    
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    # uniq=["LoF pores","Conduction mode","Keyhole pores"]
    
    z = range(1,len(uniq))
    hot = plt.get_cmap('hsv')
    
    
    
    fig = plt.figure(figsize=(20,8), dpi=100)
    
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2
    plt.rc("font", size=20)
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(elev=15,azim=110)#115
    
    
    #ax.legend(markerscale=1)
    # Plot each species
    marker= ["*",">","X","o","s"]
    color = cm.rainbow(np.linspace(0, 1, len(uniq)))
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    #ax.legend(markerscale=15)
    ax.set_ylim(min(x2), max(x2))
    ax.set_zlim(min(x3), max(x3))
    ax.set_xlim(min(x1), max(x1))
    
    for i in range(len(uniq)):
        indx = (df['label']) == uniq[i]
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        
        ax.plot(a, b, c ,color=color[i],label=uniq[i],marker=marker[i],linestyle='',ms=10)
        
        # a=x1[indx]
        # b=x2[indx]
        # c=x3[indx]
    
    plt.xlabel ('Dimension 1', labelpad=20,fontsize=15)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize=15)
    ax.set_zlabel('Dimension 3',labelpad=20,fontsize=15)
    plt.title(graph_title,fontsize = 20)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #plt.zticks(fontsize = 25)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name, bbox_inches='tight',dpi=200)
    plt.show()
    
    return ax,fig

