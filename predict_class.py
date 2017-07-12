import numpy as np
import random
import scipy.stats as ss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap as lcm
from time import strftime as stime
import os

def synthetic_plot(limits=(-5,5,-5,5), unit=0.1, no_of_points=20, no_of_classes=2, k=5):
    limits = ( limits[0], limits[1]+unit, limits[2], limits[3]+unit )
    (predictors, outcomes) = generate_synth_data(no_of_points, no_of_classes)
    
    (xx, yy, prediction_grid) = make_prediction_grid(predictors, outcomes, limits, unit, k)
    plot_prediction_grid(xx, yy, prediction_grid, predictors, outcomes);
    
    
def genuine_plot(predictors, outcomes, limits=(-5,5,-5,5), unit=0.1, k=5):
    limits = ( limits[0], limits[1]+unit, limits[2], limits[3]+unit )
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
    
    (xx, yy) = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape, dtype=int)
    
    
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = knn_predict( p , points, outcomes, k)
            
    return (xx, yy, prediction_grid)
            

def generate_synth_data(n=50,types=2):
    points = ss.norm(0 , 1).rvs((n,2))
    outcomes = np.repeat(0 , n)
    for i in range(1,types):
        points = np.concatenate( (points, ss.norm(i , 1).rvs((n,2)) ), axis=0 )
        outcomes = np.concatenate( (outcomes, np.repeat(i,n)) )
    return (points, outcomes)
    

def plot_synth_data(n=50,types=2):
    points,outcomes = generate_synth_data(n,types)
    plt.figure()
    for k in range(0,types*n,n):
        c = "#%06x" % random.randint(0, 0xFFFFFF)
        plt.plot(points[k:k+n,0], points[k:k+n,1], "o", color=c, label="Class "+str(k//n + 1))
    plt.legend(loc="upper left")
    plt.show()


def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p,points,k)
    return majority_vote(outcomes[ind])


def distance( p1=[0,0], p2=[0,0] ):
    """
    Finds the distance between two points in any dimension
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    d = np.sqrt(np.sum(np.power( p2 - p1, 2)))
    
    return d

def majority_vote(votes):
    """
    returns the maximum occurrance in a list
    """
    vote_counts = {}
       
    for vote in votes:
        if vote in vote_counts.keys():
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
            
    max_count = max(vote_counts.values())
    winners = []
    
    for (vote, count) in vote_counts.items():
        if count == max_count:
            winners.append(vote)
    
    return random.choice(winners)

def majority_vote_short(votes):
    """
    returns the mode from a list
    """
    (mode, count) = ss.mstats.mode(votes)
    return mode

def find_nearest_neighbors(p, points, k=5):
    """
    Find the k nearest neighbors of a point p
    """
    distances = np.zeros(points.shape[0])

    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
        
    ind = np.argsort(distances)

    return ind[:k]
