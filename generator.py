import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import make_classification

def generate_data(T, n, d, n_class = 4, class_sep = 10, n_redundant = 0, weights = None):
    '''
    generates the sequence of {i_t,C_t} over time.

    INPUT :
    ***** FROM THE ARTICLE   *****
    - T: number if time step
    - n: number of users
    - d: number of features

    *****  IMPLEMENTATION DETAILS   *****
    - n_class : number of latent classes in the users
    - class_sep : distance between classes
    - n_redundant: number of redundant features
    - weights: repartition of samples between the classes

    OUPUT:
    list lists, each list is the context at time t and is made of three elements:
        - indice of the user picked
        - features if the user picked
        - array of the products available


    '''

    #we introduce here different cardinality for each class
    if weights == None:
        weights = np.ones((n_class))
        if n_class > 2:
            weights[0] = 2/n_class
            weights[1:] = (n_class-2)/(n_class*(n_class-1))
    weights = weights.tolist()

    users_params = {"n_samples" : n, "n_features" : d, "n_informative" : d - n_redundant, "n_redundant" : n_redundant,
        "n_repeated" : 0, "n_classes": n_class,"n_clusters_per_class":1, "weights": weights,
        "flip_y":0.0, "class_sep": class_sep, "hypercube":None, "shift":0.0, "scale":1.0, "shuffle":True,
        "random_state":None}

    #build the matrix of users
    [users, users_labels] = make_classification(**users_params)
    users = users / (np.sum(np.abs(users)**2,axis=-1)**(1./2))[:, np.newaxis] #we normalize each row with L2 norm

    #build the matrix of product
    C = np.eye(d)

    output = []
    for t in range(T):
        i = np.random.randint(0, n)   #user picked   CHANGER DISTRIBUTION ?
        user_selected = users[i,:]

        item_number = np.random.randint(1,d)
        item_available = C[:,np.random.choice(d, item_number, replace=False)]

        output += [[i, user_selected,item_available]]

    return output
    
