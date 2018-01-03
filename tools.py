

def monitor_result(ans):
    '''

    This function take the CAB or CLUB as an argument to plot main results

    '''
    f, ax = plt.subplots(2, 2, figsize=(30,15))
    ax[0,0].plot(np.cumsum(np.array(out[2])), c='r', linewidth=2)
    ax[0,0].grid()
    ax[0,0].set_title('Cumulated regret')
    ax[0,1].plot(np.cumsum(np.array(out[2]) == 0)/np.arange(1,T+1), linewidth=2)
    ax[0,1].grid()
    ax[0,1].set_title('Proportion of perfect choices')
    ax[1,0].plot(out[6], c='g', linewidth=0.1)
    ax[1,0].grid()
    ax[1,0].set_title('Size of estimated neighborhood')
    ax[1,1].plot(out[5], c='g', linewidth=0.1)
    ax[1,1].grid()
    ax[1,1].set_title('Number of updated neighbors')


def three_D_eye_matrix(dim1,dim2):
    res = np.zeros((dim1,dim2,dim2))
    for i in range(dim1):
        res[i] = np.eye(dim2,dim2)
    return res


def build_similarity(M, sigma, fun='gauss'):
    '''
    This function build the simiarity matrix between the columns of M according to some distance
    '''
    N = M.shape[1]
    if fun == 'gauss':
        return np.exp(-squareform(pdist(np.transpose(M)))**2/(2*sigma**2))
    if fun == 'cos':
        out = np.zeros((N,N))
        for i in range(N):
            for j in range(i):
                val = cosine_similarity(M[:,i],M[:,j])
                out[i,j] = val
                out[j,i] = val
        print('similarities : ', out+ np.eye(N))
        return out + np.eye(N)


def build_graphs(W, graph_type, thresh):
    '''
    Build the graph from similarities
    '''
    if graph_type =='eps':
        W[W < thresh] == 0
        return W

    if graph_type == 'knn':
        N = W.shape[0]
        for i in range(N):
            W[i,i]=0
            idx = W[i,:].argsort()[:-thresh]
            W[i,idx]=0
        return np.maximum(W,np.transpose(W))

def SpectralClustering(Affinity, n_clusters, n_axis, nor_type):
    """
    Affinity : N by N affinity matrix, where N is the number of points.
    n_clusters : number of groups
    n_axis : number of eigenvectors to keep from the Laplacian
    nor_type : 'unn' or 'nor' for an unnormalized or normalized Laplacian
    labels : if provided, algorithm will show a 2-d spectral representation of the data,
    colored according to the labels provided

    """
    W = Affinity
    N = W.shape[0]
    D = np.zeros((N,N))

    #We build the Laplacian

    if nor_type == 'unn':
        for  i in range(N):
            D[i,i] = sum(W[i,:])

        L = D - W

    elif nor_type == 'nor':
        for  i in range(N):
            D[i,i] = sum(W[i,:])**(-1/2)

        L = np.eye(N) - D @ W @ D

    #Compute eigenvectors
    d,p = LA.eig(L)
    idx = np.argsort(d)
    d = d[idx]
    p = p[:,idx]


    Y = p[:,1:n_axis+1]

    kmeans = KMeans(n_clusters = n_clusters).fit(Y)

    return [d, Y , kmeans.labels_]


def SC(X, sigma, graph_type, thresh, n_clusters, n_axis, nor_type, labels, plot=True):
    '''
    Performs spectral clustering
    X : data
    sigma : for the gaussian similarity
    graph type: knn or eps
    thresh : eps or k
    n_clusters: number of groups
    n_axis : number of eigenvectors to keep
    labels: true label (for the plot)
    plot: whether to plot fancy illustrations or not

    '''

    W = build_similarity(X, sigma, 'gauss')
    Aff = build_graphs(W, graph_type, thresh)
    output = SpectralClustering(Aff,n_clusters, n_axis, nor_type)
    d = output[0]
    #print('Clustering error : ' , clustering_error(labels, output[2]))

    if plot == True:
        pca = PCA(n_components=2)
        proj_data = pca.fit_transform(X.T)
        proj_spectral = pca.fit_transform(output[1])

        f, ax = plt.subplots(2, 2, figsize=(30,15))
        ax[0,0].scatter(proj_data[:, 0], proj_data[:, 1], c=labels, cmap='prism', linewidths = 0.5, s=50)
        ax[0,0].set_title('Original data')
        ax[0,1].scatter(range(n_axis + 5), d[:n_axis + 5], c='b', linewidths = 1, s=75)
        ax[0,1].axvline(x = n_axis - 1/2, c='r', linewidth = 2 )
        ax[0,1].set_title('Eigenvalues')
        ax[1,0].scatter(proj_spectral[:, 0], proj_spectral[:, 1], c=labels, cmap='prism', linewidths = 0.5, s=50)
        ax[1,0].set_title('Ground truth')
        ax[1,1].scatter(proj_spectral[:, 0], proj_spectral[:, 1], c=output[2], cmap='prism', linewidths = 0.5, s=50)
        ax[1,1].set_title('K-Means')



    return output[1]
