





def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def generate_embedded_data(T, T_historical, n_users, n_products, d, d_reduced, method_users = 'blobs', method_products= 'blobs',
                  n_class_products = None, n_class_users = 4, context_len = 3, context_emb = 5, emb_noise = 0.1,
                  plot_emb = True, users_cluster_param = 0.05, products_cluster_param = 0.01, seed=None):
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
    - T_historical : number of previous timestep
    - d_reduced : the size of the embeddings
    - n_class_users : number of latent classes in the users
    - n_class_products : number of latent classes in the products
    - context_len: number of item in the context at each step
    - context_emb : size of the context for historical data
    - emb_noise : noise in the historical data
    - plot_emb : whether to plot the spectral clustering of not
    - users_cluster_param : distance between classes or std of the users clusters
    - products_cluster_param : distance between classes or std of the product clusters

    OUPUT:
    A list
        Output[0]: a matrix giving the vectors of the different users
        Output[1]: a matrix giving the vectors of the different products
        Output[2]: a matrix giving the embedded vectors of the different products
        Output[3]:  list of lists, each list is the context at time t and is made of two elements:
                    - indice of the user picked
                    - array containing indices of the available products (context)

    '''


    np.random.seed(seed)
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

    #Creation of embeddings :

    hist_mat = np.zeros((d,n_users))

    #We generate historical data
    for t in range(T_historical):
        i = np.random.randint(0, n_users)
        user_selected = users[i,:]
        idx = np.random.choice(d, context_emb, replace=False)
        items_available = products[idx,:]
        item_chosen = np.argmax(user_selected @ items_available.T + np.random.normal(0, emb_noise,(1, context_emb)))
        item_chosen = idx[item_chosen]
        hist_mat[item_chosen,i] += 1

    hist_mat = hist_mat / (np.sum(np.abs(hist_mat)**2,axis=-1)**(1./2))[:, np.newaxis]

    #We perform spectral clustering
    SC_args = {'X' : hist_mat.T, 'sigma': 100, 'graph_type':'knn', 'thresh':3 , 'n_clusters': n_class_products, 'n_axis': d_reduced,
              'nor_type':'nor', 'labels': products_labels, 'plot': plot_emb}

    embedded_products = SC(**SC_args)
    print("embedding produced")



    # Creation of the learning sequences
    output = []
    for t in range(T):
        i = np.random.randint(0, n_users)   # user picked
        item_available = np.random.choice(n_products, context_len, replace=False)
        output += [[i, item_available]]

    return [np.matrix(users), products, embedded_products, output]
