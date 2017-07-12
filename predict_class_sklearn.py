import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
from sklearn.neighbors import KNeighborsClassifier as knc
from time import strftime as stime
import os

def synthetic_plot(limits=(-5,5,-5,5), unit=0.1, no_of_points=20, no_of_classes=2, k=5):
    (predictors, outcomes) = generate_synth_data(no_of_points, no_of_classes)
    
    genuine_plot(predictors, outcomes, limits, unit, k)
    
def genuine_plot(predictors, outcomes, limits=(-5,5,-5,5), unit=0.1, k=5, home=False):
    (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, unit, k)
    
    plot_prediction_grid(xx, yy, prediction_grid, predictors, outcomes);


def plot_prediction_grid (xx, yy, predicted_grid, predictors, outcomes):
    """ Plot KNN predictions for every point on the grid."""
    
    types = len( set( outcomes ) )
    
    c_bg = np.zeros((types,3))
    c_ob = np.zeros((types,3))
    
    for i in range(types):
        c_bg_i = np.array([random.randint(100,255) / 255, random.randint(100,255) / 255, random.randint(100,255) / 255])
        c_ob_i = (c_bg_i*255 - 50)/255
        
        c_bg[i] = c_bg_i
        c_ob[i] = c_ob_i
        
    
    background_colormap = lcm(c_bg)
    observation_colormap = (c_ob)
    
    plt.figure( figsize =(10,10) )
    
    plt.pcolormesh(xx, yy, predicted_grid, cmap = background_colormap, alpha = 0.5)
    
    xs = np.array(predictors[:,0])
    ys = np.array(predictors[:,1])
    outcomes = np.array( outcomes )
    
    
    for i in range(types):
        to_plot = outcomes==i
        plt.scatter(xs[to_plot] , ys[to_plot] ,s = 50,color=observation_colormap[i] , label="Class "+str(i+1))
    
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    
    x_labels = np.linspace( np.min(xx), np.max(xx), 5 )
    y_labels = np.linspace( np.min(yy), np.max(yy), 5 )
    
    plt.xticks(x_labels, rotation="vertical")
    plt.yticks(y_labels)
    
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    
    plt.legend(loc="lower right")
    
    if not os.path.exists("Plots"):
        os.makedirs("Plots")
    
    filename = "Plots\plot_" + stime("%d-%m-%Y_%H-%M-%S") + ".pdf"
    plt.savefig(filename)
    plt.show()

def make_prediction_grid(points, outcomes, limits, steps=1, k=5):
    (x_min, x_max, y_min, y_max) = limits
    xs = np.arange(x_min, x_max, steps)
    ys = np.arange(y_min, y_max, steps)
    
    
    knn = knc(n_neighbors=k)
    knn.fit(points,outcomes)
    
    (xx, yy) = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    
    
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn.predict([p])[0]
                
    return (xx, yy, prediction_grid)
            

def generate_synth_data(n=50,types=2):
    points = ss.norm(0 , 1).rvs((n,2))
    outcomes = np.repeat(0 , n)
    for i in range(1,types):
        points = np.concatenate( (points, ss.norm(i , 1).rvs((n,2)) ), axis=0 )
        outcomes = np.concatenate( (outcomes, np.repeat(i,n)) )
    return (points, outcomes)


