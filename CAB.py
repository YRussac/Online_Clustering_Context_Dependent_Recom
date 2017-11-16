import numpy as np
import math
from numpy.linalg import inv
from generator import generate_data


def Context_Aware_Clustering_of_Bandits(gamma, alpha, d, n, T):
    '''

    :param gamma:
    :param alpha:
    :param d:
    :param n:
    :param T:
    :return:
    '''

    b_list = [np.matrix(np.zeros(d)).T for i in range(n)]
    M_list = [np.eye(d) for i in range(n)]
    generation_tot = generate_data(T=T, n=n, d=d)
    data_generation = generation_tot[1]
    users_info = generation_tot[0]
    CB= np.zeros((n, d))  # matrix i: user j: numero de l'item
    for t in range(T):
        # filling the CB Matrix
        for i in range(n):
            for k in range(d):
                x = np.zeros(d)
                x[k] = 1
                x = np.matrix(x).T
                CB[i, k] = alpha[t]*math.sqrt(x.T*inv(M_list[i])*x)
        # The CB Matrix is full
        w_list = [inv(M_list[i])*b_list[i] for i in range(n)]
        i_t, u_i_t, C_t = data_generation[t]
        u_i_t = np.matrix(u_i_t).T
        K = C_t.shape[1]  # number of items in the context
        dico_t = {}  # for each item will contain list of neighbors
        for k in range(K):
            num_item = np.where(C_t[:, k] == 1.)[0][0]  # find the 1 in the One-Hot vector
            neigh_k_it = []
            w_dico_N_K = {}
            CB_dico_N_K= {}
            # Creation of the neighborhood for each item
            for j in range(n):
                if j != i_t:
                    if abs(w_list[i_t].T*C_t[:, k] - w_list[j].T*C_t[:, k])[0, 0] <= CB[i_t, num_item] + CB[j, num_item]:
                            neigh_k_it.append(j)
            sum_calc = sum([w_list[neigh] for neigh in neigh_k_it])
            sum_calc_CB = sum([CB[j, num_item] for j in neigh_k_it])
            w_dico_N_K[num_item] = sum_calc/len(neigh_k_it)
            CB_dico_N_K[num_item] = sum_calc_CB/len(neigh_k_it)
            dico_t[num_item] = neigh_k_it
        # Part for the item recommendation
        maxi = - math.inf
        indice_contexte = 0
        for k in range(K):
            num_item = np.where(C_t[:, k] == 1.)[0][0]
            if num_item in w_dico_N_K.keys():
                a = (w_dico_N_K[num_item].T*C_t[:, k] + CB_dico_N_K[num_item])[0, 0]
                if a >= maxi:
                    maxi = a
                    indice_contexte = k
        item_propos = np.where(C_t[:, indice_contexte] == 1.)[0][0]
        y_t = (u_i_t.T*C_t[:, indice_contexte])[0, 0] + np.random.normal(0,0.1) # check std here
        if CB[i_t, item_propos] >= gamma/4:
            M_list[i_t] = M_list[i_t] + C_t[:, indice_contexte]*C_t[:, indice_contexte].T
            b_list[i_t] = b_list[i_t] + y_t*C_t[:, indice_contexte]
        else:
            for neigh in dico_t[item_propos]:
                if CB[neigh, item_propos] < gamma/4:
                    M_list[neigh] = M_list[neigh] + C_t[:, indice_contexte]*C_t[: indice_contexte]
                    b_list[neigh] = b_list[neigh] + y_t*C_t[:, indice_contexte]
    return [users_info, w_list]