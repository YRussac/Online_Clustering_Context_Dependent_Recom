def CLUB_general(T, alpha, alpha_2, n_users, n_products, d_large, embedding_param = None,
                 n_class_users = 4, n_class_products = None, context_len = 3,
                 payoff_noise = 0.001, users_cluster_param = 0.05,
                 products_cluster_param = 0.01, method_users = 'blobs',
                 method_products= 'One-Hot',
                 ):
    '''
    Implementation of the CLUB algorithm presented in Online Clustering of Bandits (Gentille & al.)

     INPUT :
    - T: number of period for the algorithm: int
    - alpha: parameter used to calculate the standard confident bound (CB) for different users
     for the different periods. (exploration parameter): int
    - alpha_2: parameter used for the graph update: int
    - n_users: number of users: int
    - n_products: number of products: int
    - d_large: dimension of the user representation: int
    - embedding_param : if None: regular CLUB is performed.
                        if list [number of hidden class of products, size of the embedding],
                        an embedded CAB is performed
    - n_class_users : number of underlying user clusters: int
    - n_class_products : number of underlying product clusters: int
    - context_len : number of item available at each timestep: int
    - payoff_noise : std for the noise of the payoff_noise: int
    - user_cluster_param: distance between classes or std of the users clusters: int
    - product_cluster_param: distance between classes or std of the product clusters: int
    - method_users: string giving the method for the generation of the users: str
    - method_products: string giving the method for the generation of the products: str

    OUTPUT: users_matrix, w_matrix, regret, choices, best_choices, cluster_dico
    - users_matrix: matrix containing the real users vectors: matrix
    - w_matrix: proxy vectors for users: matrix
    - regret: regret at each timestep
    - choices: choice made by the model at each time step
    - best_choices: best choice at each timestep
    - cluster_dico: dictionnary where each cluster is mapped to the users belonging to the cluster
    '''



    # Initialisation of the different parameters
    regret = []
    choices = []
    best_choices = []
    m_t = 1
    adjacency_matrix = np.ones((n_users, n_users))
    CB_tilde = np.zeros(shape=(n_users, 1))
    occurence_users = [0]*n_users
    cluster_dico = {}
    cluster_dico[0] = [i for i in range(n_users)]

    if embedding_param:
        n_class_products, d_reduced, T_historical, context_emb, emb_noise, plot_emb = embedding_param
        b_matrix = np.zeros(shape=(n_users, d_reduced, 1))
        M_matrix = three_D_eye_matrix(n_users, d_reduced)
        generation_data = generate_embedded_data(T, T_historical, n_users, n_products, d, d_reduced, method_users, method_products,
                          n_class_products, n_class_users, context_len, context_emb, emb_noise,
                          plot_emb, users_cluster_param, products_cluster_param )
        (users_matrix, original, products, data_generation) = (generation_data[0], generation_data[1], generation_data[2],generation_data[3])
        w_matrix = np.zeros(shape = (n_users,d_reduced,1))


    else:
        b_matrix = np.zeros(shape=(n_users, d_large, 1))
        M_matrix = three_D_eye_matrix(n_users, d_large)
        generation_data = generate_data(T, n_users, n_products, d_large, method_users, method_products,
                          n_class_products, n_class_users, context_len,
                          users_cluster_param, products_cluster_param)
        (users_matrix, products, data_generation) = (generation_data[0], generation_data[1], generation_data[2])
        original = products
        w_matrix = np.zeros(shape=(n_users, d_large, 1))


    # Sequential learning
    for t in range(T):
        if t % 500 == 0:
            print("##################")
            print('tour n° ' + str(t))
            print("##################")

        for i in range(n_users):
            w_matrix[i] = inv(M_matrix[i]) @ b_matrix[i]

        CB = np.zeros((m_t, n_products))
        i_t, C_t = data_generation[t]
        u_i_t = users_matrix[i_t, :].T
        K = len(C_t)  # number of items in the context
        if t == 0:
            j_t = 0
        else:
            j_t = cluster_indices[i_t]  # cluster to which i_t belongs
        M_mean = np.zeros(shape=(n_products, n_products))
        b_mean = np.zeros(shape=(n_products, 1))

        for indice in cluster_dico[j_t]:
            M_mean += M_matrix[indice] - np.eye(n_products, n_products)
            b_mean += b_matrix[indice]

        M_mean = M_mean + np.eye(n_products, n_products)
        w_mean = inv(M_mean)@b_mean

        maxi = - math.inf
        indice_contexte = 0
        for k in range(K):
            num_item = C_t[k]

            CB[j_t, num_item] = alpha * math.sqrt((products[num_item, :] @ inv(M_matrix[j_t])
                                                @ products[num_item, :].T) * math.log(t+1))
            temp = w_mean.T@(products[num_item, :][:, np.newaxis]) + CB[j_t, num_item]
            if temp >= maxi:
                maxi = temp
                indice_contexte = k

        # Part for the item recommendation
        item_propos = C_t[indice_contexte]
        choices += [item_propos]
        # We compute the linear payoff
        a_t = float(u_i_t.T @ original[item_propos, :] + np.random.normal(0, payoff_noise))

        # We compute regret
        best_choice = np.argmax(u_i_t.T @ original[C_t,:].T)
        best_choices += [C_t[best_choice]]
        regret += [float(u_i_t.T @ original[C_t[best_choice],:].T -
                         u_i_t.T @ original[item_propos,:].T)]


        # Updating part M and b matrix
        M_matrix[i_t] = M_matrix[i_t] + products[C_t[indice_contexte], :][:, np.newaxis] @ \
                                        products[C_t[indice_contexte], :][:, np.newaxis].T
        b_matrix[i_t] = b_matrix[i_t] + a_t * products[C_t[indice_contexte],:][:, np.newaxis]

        for l in range(n_users):
            CB_tilde[l, 0] = alpha_2 * math.sqrt((1+math.log(1+occurence_users[l])) /
                                                 (1+occurence_users[l]))
            if math.sqrt((w_matrix[i_t]-w_matrix[l]).T@
                         (w_matrix[i_t]-w_matrix[l])) > (CB_tilde[i_t, 0] + CB_tilde[l, 0]):
                adjacency_matrix[i_t, l] = 0
                adjacency_matrix[l, i_t] = 0

        adjacency_nx = nx.Graph(adjacency_matrix)
        connected_comp = nx.connected_components(adjacency_nx)

        cluster_dico = {}
        for nb_cluster, cluster_member in enumerate(connected_comp):
            cluster_dico[nb_cluster] = cluster_member
        m_t = len(list(cluster_dico.keys()))
        cluster_indices = {}
        for i in range(len(cluster_dico.keys())):
            l = cluster_dico[i]
            for node in l:
                cluster_indices[node] = i

        occurence_users[i_t] += 1

    return [users_matrix, w_matrix, regret, choices, best_choices, cluster_dico]
