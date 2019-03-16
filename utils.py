import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import math

flags = tf.app.flags
FLAGS = flags.FLAGS


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num, ent2id):
    """Load a file and return a list of tuple containing $num integers in each line."""
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.strip().split('\t')
            x = []
            for i in range(num):
                x.append(ent2id[th[i]])
            ret.append(tuple(x))
    return ret


def loadattr(fns, e, ent2id):
    """The most frequent attributes are selected to save space."""
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, 2):
                    th[i] = th[i].split('/')[-1]
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, 2):
                        th[i] = th[i].split('/')[-1]
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {}
    for i in range(e):
        dic_list[i] = []
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))


def load_KG(Rs):
    ent2id = {}
    e = 0
    rel2id = {}
    r = 0
    KG = []
    for fn in Rs:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                if th[0] not in ent2id:
                    ent2id[th[0]] = e
                    e += 1
                if th[1] not in rel2id:
                    rel2id[th[1]] = r
                    r += 1
                if th[2] not in ent2id:
                    ent2id[th[2]] = e
                    e += 1
                KG.append((ent2id[th[0]], rel2id[th[1]], ent2id[th[2]]))
    return ent2id, KG


def load_data(dataset_str):
    names = [['s_triples', 't_triples'], ['s_triples_attr', 't_triples_attr'], ['ent_ILLs']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'dbp15k/'+dataset_str+'/'+fns[i]
    Rs, As, ill = names
    ill = ill[0]
    ent2id, KG = load_KG(Rs)
    e = len(ent2id)
    ILL = loadfile(ill, 2, ent2id)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * FLAGS.seed])
    test = ILL[illL // 10 * FLAGS.seed:]
    attr = loadattr(As, e, ent2id)
    ae_input = get_ae_input(attr)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG)))
    return adj, ae_input, train, test

