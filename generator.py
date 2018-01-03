import numpy as np
from sklearn.datasets import make_classification, make_blobs


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def generate_data(T, n_users, n_products, d, method_users = 'blobs', method_products= 'blobs',
                  n_class_products = None, n_class_users = 4, context_len = 3,
                  users_cluster_param = 0.05, products_cluster_param = 0.01):
    '''
    generates the sequence of {i_t,C_t} over time.


    INPUT :
    ***** FROM THE ARTICLE   *****
    - T: number if time step
    - n_users: number of users
    - n_products: number of products
    - d: number of features
    - method_users: string giving the method for the generation of the users
    - method_products: string giving the method for the generation of the products


    *****  IMPLEMENTATION DETAILS   *****
    - n_class_users : number of latent classes in the users
    - n_class_products : number of latent classes in the products
    - context_len: number of item in the context at each step
    - users_cluster_param : distance between classes or std of the users clusters
    - products_cluster_param : distance between classes or std of the product clusters

    OUPUT:
    A list
        Output[0]: a matrix giving the vectors of the different users
        Output[1]: a matrix giving the vectors of the different products
        Output[2]:  list of lists, each list is the context at time t and is made of two elements:
                    - indice of the user picked
                    - array containing indices of the available products (context)

    '''

    # Create the users
    if method_users == 'blobs':
        users_params = {"n_samples": n_users, "n_features": d, "centers": sample_spherical(n_class_users, d).T,
                        "cluster_std": users_cluster_param, "shuffle": True, "random_state": None}
        [users, users_labels] = make_blobs(**users_params)

    elif method_users == 'classification':
        users_params = {"n_samples": n_users, "n_features": d, "n_informative": d,
                        "n_repeated": 0, "n_classes": n_class_users, "n_clusters_per_class": 1,
                        "flip_y": 0.0, "class_sep": users_cluster_param, "hypercube": None,
                        "shift": 0.0, "scale": 1.0, "shuffle": True,
                        "random_state": None}
        [users, users_labels] = make_classification(**users_params)
    else:
        raise ValueError('Unknown method used for the generation of the users')

    users = users / (np.sum(np.abs(users)**2, axis=-1)**(1./2))[:, np.newaxis]

    # Create the products
    if method_products == 'blobs':
        if n_class_products:
            products_params = {"n_samples": n_products, "n_features": d,
                               "centers": sample_spherical(n_class_products, d).T,
                               "cluster_std": products_cluster_param, "shuffle": True, "random_state": None}
            [products, products_labels] = make_blobs(**products_params)
        else:
            raise ValueError('n_class_products should be specified to use the blobs method')

    elif method_products == 'classification':
        if n_class_products:
            products_params = {"n_samples": n_products, "n_features": d,  "n_informative": d,
                               "n_repeated": 0, "n_classes": n_class_products, "n_clusters_per_class": 1,
                               "flip_y": 0.0, "class_sep": products_cluster_param, "hypercube": None,
                               "shift": 0.0, "scale": 1.0, "shuffle": True, "random_state": None}
            [products, products_labels] = make_classification(**products_params)
        else:
            raise ValueError('n_class_products should be specified to use the classification method')

    elif method_products == 'One-Hot':
        if d != n_products:
            raise ValueError('When using One-Hot d and n_products should be equal')
        else:
            products = np.eye(d)

    else:
        raise ValueError('Unknown method used for the generation of the products')

    products = products / (np.sum(np.abs(products)**2, axis=-1)**(1./2))[:, np.newaxis]

    # Creation of the learning sequences
    output = []
    for t in range(T):
        i = np.random.randint(0, n_users)   # user picked
        item_available = np.random.choice(n_products, context_len, replace=False)
        output += [[i, item_available]]

    return [np.matrix(users), np.matrix(products), output]
