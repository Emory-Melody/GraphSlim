import torch


def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def distance_wb(gwr, gws):
    shape = gwr.shape

    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


# GCSNTK utils
def sub_E(idx, A):
    """
    output the sparse adjacency matrix of subgraph of idx
    """
    n = A.shape[0]
    n_neig = len(idx)
    operator = torch.zeros([n, n_neig])
    operator[idx, range(n_neig)] = 1
    sub_A = torch.matmul(torch.matmul(operator.t(), A), operator)

    ind = torch.where(sub_A != 0)
    inds = torch.cat([ind[0], ind[1]]).reshape(2, len(ind[0]))
    values = torch.ones(len(ind[0]))
    sub_E = torch.sparse_coo_tensor(inds, values, torch.Size([n_neig, n_neig])).to(A.device)

    return sub_E


def update_E(x_s, neig):
    '''
    x_s is the features
    neig is the average number of the neighbors of each node
    '''
    n = x_s.shape[0]
    K = torch.empty(n, n)
    A = torch.zeros(n * n)

    for i in range(n):
        for j in range(i, n):
            K[i, j] = torch.norm(x_s[i] - x_s[j])
            K[j, i] = K[i, j]

    edge = int(n + torch.round(torch.tensor(neig * n / 2)))
    if (edge % 2) != 0:
        edge += 1
    else:
        pass

    Simil = torch.flatten(K)
    _, indices = torch.sort(Simil)
    A[indices[0:edge]] = 1
    A = A.reshape(n, n)
    ind = torch.where(A == 1)

    ind = torch.cat([ind[0], ind[1]]).reshape(2, edge)
    values = torch.ones(edge)
    E = torch.sparse_coo_tensor(ind, values, torch.Size([n, n])).to(x_s.device)

    return E


# utils for GCSNTK
def normalize_data(data):
    """
    normalize data
    parameters:
        data: torch.Tensor, data need to be normalized
    return:
        torch.Tensor, normalized data
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    std[std == 0] = 1
    normalized_data = (data - mean) / std
    return normalized_data


def GCF(adj, x, k=1):
    """
    Graph convolution filter
    parameters:
        adj: torch.Tensor, adjacency matrix, must be self-looped
        x: torch.Tensor, features
        k: int, number of hops
    return:
        torch.Tensor, filtered features
    """
    D = torch.sum(adj, dim=1)
    D = torch.pow(D, -0.5)
    D = torch.diag(D)

    filter = torch.matmul(torch.matmul(D, adj), D)
    for i in range(k):
        x = torch.matmul(filter, x)
    return x
