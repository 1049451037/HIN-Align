import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import math
import os

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


def loadfile(fn, num, ent2id=None):
    """Load a file and return a list of tuple containing $num integers in each line."""
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line.strip().split('\t')
            x = []
            for i in range(num):
                if ent2id is None:
                    x.append(int(th[i]))
                else:
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
                    #th[i] = th[i].split('/')[-1]
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
                        #th[i] = th[i].split('/')[-1]
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


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = r2if[tri[1]]
        else:
            M[(tri[0], tri[2])] += r2if[tri[1]]
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = r2f[tri[1]]
        else:
            M[(tri[2], tri[0])] += r2f[tri[1]]
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(sigmoid(M[key]))
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def load_KG(Rs):
    ent2id_div = [{}, {}]
    ent2id = {}
    e = 0
    rel2id = {}
    r = 0
    KG = [[], []]
    for i in range(len(Rs)):
        fn = Rs[i]
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip().split('\t')
                if th[0] not in ent2id:
                    ent2id[th[0]] = e
                    ent2id_div[i][th[0]] = e
                    e += 1
                if th[1] not in rel2id:
                    rel2id[th[1]] = r
                    r += 1
                if th[2] not in ent2id:
                    ent2id[th[2]] = e
                    ent2id_div[i][th[2]] = e
                    e += 1
                KG[i].append((ent2id[th[0]], rel2id[th[1]], ent2id[th[2]]))
    return ent2id, ent2id_div, KG


def load_sn_data():
    names = [['foursquare', 'twitter'], ['link']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'IONE_data/'+fns[i]
    Rs, ill = names
    ill = ill[0]
    HIN = [loadfile(Rs[0], 2), loadfile(Rs[1], 2)]
    link = loadfile(ill, 1)
    ids0 = set()
    for x, y in HIN[0]:
        ids0.add(x)
        ids0.add(y)
    ids1 = set()
    for x, y in HIN[1]:
        ids1.add(x)
        ids1.add(y)
    KG = [[], []]
    ILL = []
    for i in range(2):
        for x, y in HIN[i]:
            KG[i].append((x+i*len(ids0), i, y+i*len(ids0)))
    for x in link:
        ILL.append((x[0], x[0]+len(ids0)))
    e = len(ids0) + len(ids1)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * FLAGS.seed])
    test = ILL[illL // 10 * FLAGS.seed:]
    adj = get_weighted_adj(e, KG[0]+KG[1]) # nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG[0]+KG[1])))
    return adj, train, test, KG, e


def load_data(dataset_str):
    names = [['s_triples', 't_triples'], ['s_triples_attr', 't_triples_attr'], ['ent_ILLs']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'dbp15k/'+dataset_str+'/'+fns[i]
    Rs, As, ill = names
    ill = ill[0]
    ent2id, ent2id_div, KG = load_KG(Rs)
    e = len(ent2id)
    ILL = loadfile(ill, 2, ent2id)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * FLAGS.seed])
    test = ILL[illL // 10 * FLAGS.seed:]
    attr = loadattr(As, e, ent2id)
    ae_input = get_ae_input(attr)
    adj = get_weighted_adj(e, KG[0]+KG[1]) # nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG[0]+KG[1])))
    return adj, ae_input, train, test, ent2id_div, KG


def jape_results_to_gcn(mp1, mp2, embeddings, saved_filepath):
    gb_list = []
    for key, value in mp1.items():
        gb_list.append([key, value])
    for key, value in mp2.items():
        gb_list.append([key, value])
    gb_np = np.array(sorted(gb_list))
    assert len(set(gb_np.T[0])) == len(mp1)+len(mp2)
    np.save(saved_filepath, embeddings[gb_np.T[1]])
    return


def gcn_data_to_jape(merged, align, e1, e2, KG1, KG2, ratio_str, out_path):
    out_path_truely = out_path+ratio_str.replace('.', '_')+'/'
    if not os.path.exists(out_path_truely):
        os.makedirs(out_path_truely)
    mp1, mp2 = {}, {}
    n_merged = len(merged)
    n_align = len(align)

    t = 0
    for line in merged:
        mp1[line[0]] = 2*n_align+t
        mp2[line[1]] = 2*n_align+t
        t = t+1
    t = 0
    for line in align:
        mp1[line[0]] = t
        mp2[line[1]] = t+n_align
        t = t+1

    print("check: ", len(mp1), len(mp2))

    st = n_align*2+n_merged
    for e in e1.items():
        if e[1] not in mp1:
            mp1[e[1]] = st
            st = st+1
    st = len(e1)+n_align
    for e in e2.items():
        if e[1] not in mp2:
            mp2[e[1]] = st
            st = st+1

    print("check: ", len(mp1), len(mp2))

    with open(out_path_truely+'sup_ent_ids', 'w', encoding='utf-8') as f:
        for i in range(2*n_align, 2*n_align+n_merged):
            f.write(str(i)+'\t'+str(i)+'\n')
    open(out_path_truely+'sup_rel_ids', 'w', encoding='utf-8').close()
    with open(out_path_truely + 'triples_1', 'w', encoding='utf-8') as f:
        for line in KG1:
            f.write(str(mp1[line[0]])+'\t'+str(line[1])+'\t'+str(mp1[line[2]])+'\n')
    with open(out_path_truely + 'triples_2', 'w', encoding='utf-8') as f:
        for line in KG2:
            f.write(str(mp2[line[0]])+'\t'+str(line[1])+'\t'+str(mp2[line[2]])+'\n')
    with open(out_path_truely + 'ref_ent_ids', 'w', encoding='utf-8') as f:
        for i in range(n_align):
            f.write(str(i)+'\t'+str(i+n_align)+'\n')
    return mp1, mp2
