def CAB_general(T, gamma, alpha, n_users, n_products, d_large, embedding_param = None,
                n_class_users = 4, bound = 3, payoff_noise = 0.001,
                method_users = 'blobs', method_products= 'blobs',
                n_class_products = None, n_class_users = 4, context_len = 3,
                users_cluster_param = 0.05, products_cluster_param = 0.01):
    '''
    Implementation of the CAB algorithm presented in On Context-Dependent Clustering of Bandits
    INPUT:
    T : number of time step : int
    gamma: gap parameter : float
    alpha: exploration function : list of length n
    n_users : number of users
    n_products : number of existing products
    d_large : dimension of the user representation
    embedding_param : if None: regular CAB is performed.
                    If list, an embedded CAB is performed, in the list:
                        - n_class_products : number of underlying clusters for products
                        - d_reduced : size of the embeddings
                        - T_historical : duration of historical data
                        - context_emb : size of the context for historical data
                        - emb_noise : noise of the historical data
                        - plt_emb : whether to plot the SC or not
    n_class_users : number of underlying user clusters
    bound : number of item available at each timestep
    payoff_noise : std for the noise of the payoff_noise

    OUTPUT: a list of:
    users_matrix : matrix of the users
    w_matrix : matrix of the proxys
    regret : regret at each timestep
    choices : choice made by the model at each time step
    best_choices : best choice at each timestep
    updates : number of updated neighbors at each timestep
    neighbors_story : number of estimated neighbors at each time step


    '''

    if len(alpha) != T:
        raise ValueError('alpha should have a length = T') #if exploration is not provided at each timestep

    # to monitor the evolution of the model
    regret = []
    choices= []
    best_choices =[]
    updates = []
    neighbors_story = []


    # We generate all the data for the experiment
    if embedding_param:
        n_class_products, d_reduced, T_historical, context_emb, emb_noise, plot_emb = embedding_param
        b_matrix = np.zeros(shape = (n,d_reduced,1))
        M_matrix = three_D_eye_matrix(n,d_reduced)
        generation_data = generate_embedded_data(T, T_historical, n_users, n_products, d, d_reduced, method_users, method_products,
                          n_class_products, n_class_users, context_len, context_emb, emb_noise,
                          plot_emb, users_cluster_param, products_cluster_param )
        (users_matrix, products, original, data_generation) = (generation_data[0], generation_data[1], generation_data[2],generation_data[3])
        w_matrix = np.zeros(shape = (n,d_reduced,1))

    else:
        b_matrix = np.zeros(shape = (n,d_large,1))
        M_matrix = three_D_eye_matrix(n,d_large)
        generation_data = generate_data(T, n_users, n_products, d_large, method_users, method_products,
                          n_class_products, n_class_users, context_len,
                          users_cluster_param, products_cluster_param)
        (users_matrix, products,  data_generation) = (generation_data[0], generation_data[1], generation_data[2])
        original = products
        w_matrix = np.zeros(shape = (n,d_large,1))


    #Initialization of confidence bound
    CB = np.zeros((n_users, n_products))


    #Now we launch the model
    for t in range(T):
        if t%500 ==0 :
            print("##################")
            print('tour n° ' + str(t))
            print("##################")

        for i in range(n):
            w_matrix[i] = inv(M_matrix[i]) @ b_matrix[i] #update the user's proxy

        #We receive user and context
        i_t, C_t = data_generation[t]
        u_i_t = users_matrix[i_t,:].T
        K = len(C_t)  # number of items in the context

        #We compute solution
        dico_t = {}  # estimated neighborhoods
        expected_reward = []

        for k in range(K):
            num_item = C_t[k]
            neigh_k_it = []

            # Creation of the neighborhood for each item
            for j in range(n):
                CB[j, num_item] = alpha[t]*math.sqrt(products[num_item,:] @ inv(M_matrix[j]) @ products[num_item,:].T)
                if abs(w_matrix[i_t].T @ products[num_item,:] - w_matrix[j].T @ products[num_item,:])[0] <= CB[i_t, num_item] + CB[j, num_item]:
                        neigh_k_it.append(j)

            w = sum([w_matrix[neigh] for neigh in neigh_k_it])/len(neigh_k_it) #average proxy
            CB = sum([CB[j, num_item] for j in neigh_k_it])/len(neigh_k_it) #average CB
            dico_t[num_item] = neigh_k_it #estimated neighborhood
            expected_reward += [(w.T @ products[num_item,:]+ CB)[0]]

            neighbors_story += [len(neigh_k_it)]

        # Part for the item recommendation
        indice_contexte = np.argmax(expected_reward)
        item_propos = C_t[indice_contexte]  # this is the item that will be proposed
        choices += [item_propos]

        #We compute linear payoff
        y_t = float(u_i_t.T @ original[item_propos,:] + np.random.normal(0, payoff_noise))

        #We compute regret
        best_choice = np.argmax(u_i_t.T @ original[C_t,:].T)
        best_choices += [C_t[best_choice]]
        regret += [float(u_i_t.T @ original[C_t[best_choice],:].T - u_i_t.T @ original[item_propos,:].T)]

        #Update of the proxys
        if CB[i_t, item_propos] >= gamma/4:
            M_matrix[i_t] = M_matrix[i_t] + products[C_t[indice_contexte],:][:,np.newaxis] @ products[C_t[indice_contexte],:][:,np.newaxis].T
            b_matrix[i_t] = b_matrix[i_t] + y_t * products[C_t[indice_contexte],:][:,np.newaxis]
            updates += [1]

        else:
            up = 0
            for neigh in dico_t[item_propos]:
                if CB[neigh, item_propos] < gamma/4:
                    M_matrix[neigh] = M_matrix[neigh] + products[C_t[indice_contexte],:][:,np.newaxis] @ products[C_t[indice_contexte],:][:,np.newaxis].T
                    b_matrix[neigh] = b_matrix[neigh] + y_t*products[C_t[indice_contexte],:][:,np.newaxis]
                    up += 1
            updates += [up]

    return [users_matrix, w_matrix, regret, choices, best_choices, updates, neighbors_story]
