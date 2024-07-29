from graphslim.utils import *
import torch_geometric
import math

def match_loss(gw_syn, gw_real, args, device):
    """
    Computes the loss between synthetic and real gradients based on the specified distance metric.

    Parameters
    ----------
    gw_syn : list of torch.Tensor
        List of synthetic gradients for different model parameters.
    gw_real : list of torch.Tensor
        List of real gradients for different model parameters.
    args : Namespace
        Arguments object containing hyperparameters for training and model.
    device : torch.device
        Device (CPU or GPU) on which computations are performed.

    Returns
    -------
    torch.Tensor
        The computed distance (loss) between synthetic and real gradients.
    """
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
    """
    Computes the distance between two tensors representing gradients using cosine similarity.

    Parameters
    ----------
    gwr : torch.Tensor
        The real gradient tensor.
    gws : torch.Tensor
        The synthetic gradient tensor.

    Returns
    -------
    torch.Tensor
        The computed distance between the real and synthetic gradients.
    """
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
    Generates a sparse adjacency matrix of the subgraph defined by the given indices.

    Parameters
    ----------
    idx : torch.Tensor
        A tensor containing the indices of the nodes that define the subgraph.
    A : torch.Tensor
        The original adjacency matrix of the graph.

    Returns
    -------
    torch.sparse_coo_tensor
        The sparse adjacency matrix of the subgraph.
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
    """
    Update the adjacency matrix based on the features of the nodes and the average number of neighbors.

    Parameters
    ----------
    x_s : torch.Tensor
        A tensor containing the feature vectors of the nodes.
    neig : float
        The average number of neighbors each node should have.

    Returns
    -------
    torch.sparse_coo_tensor
        The sparse adjacency matrix based on the updated similarities.
    """
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
    Normalize the input data using mean and standard deviation.

    Parameters
    ----------
    data : torch.Tensor
        The data to be normalized. Each column represents a feature, and normalization is applied to each feature independently.

    Returns
    -------
    torch.Tensor
        The normalized data where each feature has zero mean and unit variance.
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    std[std == 0] = 1
    normalized_data = (data - mean) / std
    return normalized_data


def GCF(adj, x, k=1):
    """
    Apply Graph Convolution Filter (GCF) to features using the adjacency matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Adjacency matrix of the graph. It must include self-loops. Shape: (N, N), where N is the number of nodes.

    x : torch.Tensor
        Node features. Shape: (N, F), where F is the number of features for each node.

    k : int, optional
        Number of hops (or layers) to apply the filter. Default is 1.

    Returns
    -------
    torch.Tensor
        Filtered features after applying the graph convolution. Shape: (N, F).
    """
    D = torch.sum(adj, dim=1)
    D = torch.pow(D, -0.5)
    D = torch.diag(D)

    filter = torch.matmul(torch.matmul(D, adj), D)
    for i in range(k):
        x = torch.matmul(filter, x)
    return x


# geom
def neighborhood_difficulty_measurer(data, adj, label):
    """
    Measure the difficulty of neighborhoods in the graph based on the label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph. The shape is (N, N) where N is the number of nodes.

    label : torch.Tensor
        Tensor containing the label of each node. Shape: (N,)

    Returns
    -------
    torch.Tensor
        Difficulty scores for each node. Higher scores indicate more difficult neighborhoods.
    """
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = torch_geometric.utils.add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = neighbor_class[data.idx_train]
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    return local_difficulty


def difficulty_measurer(data, adj, label):
    """
    Measure the difficulty of nodes in the graph based on their neighborhood label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph. The shape is (N, N) where N is the number of nodes.

    label : torch.Tensor
        Tensor containing the label of each node. Shape: (N,)

    Returns
    -------
    torch.Tensor
        Difficulty scores for each node. Higher scores indicate more difficult nodes.
    """
    local_difficulty = neighborhood_difficulty_measurer(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes(data, adj, label):
    """
    Sort training nodes based on their difficulty measured by neighborhood label distribution.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    numpy.ndarray
        Indices of the training nodes sorted by their difficulty, from easiest to hardest.
    """
    node_difficulty = difficulty_measurer(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()
    sorted_trainset = data.idx_train[indices]
    return sorted_trainset


def neighborhood_difficulty_measurer_in(data, adj, label):
    """
    Measure the difficulty of each node in a graph based on the entropy of neighbor labels.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    torch.Tensor
        Tensor of local difficulty scores for each node.
    """
    edge_index = adj.coalesce().indices()
    edge_value = adj.coalesce().values()

    neighbor_label, _ = torch_geometric.utils.add_self_loops(edge_index)  # [[1, 1, 1, 1],[2, 3, 4, 5]]

    neighbor_label[1] = label[neighbor_label[1]]  # [[1, 1, 1, 1],[40, 20, 19, 21]]

    neighbor_label = torch.transpose(neighbor_label, 0, 1)  # [[1, 40], [1, 20], [1, 19], [1, 21]]

    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)

    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()

    neighbor_class = F.normalize(neighbor_class, 1.0, 1)
    neighbor_entropy = -1 * neighbor_class * torch.log(neighbor_class + torch.exp(torch.tensor(-20)))  # 防止log里面是0出现异常
    local_difficulty = neighbor_entropy.sum(1)

    return local_difficulty


def difficulty_measurer_in(data, adj, label):
    """
    Measure the difficulty of each node in a graph based on local entropy of neighbor labels.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    torch.Tensor
        Tensor of local difficulty scores for each node.
    """
    local_difficulty = neighborhood_difficulty_measurer_in(data, adj, label)
    # global_difficulty = feature_difficulty_measurer(data, label, embedding)
    node_difficulty = local_difficulty
    return node_difficulty


def sort_training_nodes_in(data, adj, label):
    """
    Sort training nodes based on their difficulty scores in ascending order.

    Parameters
    ----------
    data : Data
        PyG Data object containing node features and labels.

    adj : torch.Tensor
        Sparse adjacency matrix of the graph (shape: N x N) with self-loops.

    label : torch.Tensor
        Tensor containing the label of each node (shape: N,).

    Returns
    -------
    numpy.ndarray
        Indices of training nodes sorted by difficulty scores.
    """
    node_difficulty = difficulty_measurer_in(data, adj, label)
    _, indices = torch.sort(node_difficulty)
    indices = indices.cpu().numpy()
    return indices

def training_scheduler(lam, t, T, scheduler='geom'):
    """
    Adjust the value of a parameter based on the chosen scheduling strategy.

    Parameters
    ----------
    lam : float
        The initial value or a baseline value for the parameter (0 <= lam <= 1).

    t : int
        The current training iteration or epoch.

    T : int
        The total number of training iterations or epochs.

    scheduler : str, optional
        The type of scheduling strategy to use. Options are 'linear', 'root', or 'geom'.
        Default is 'geom'.

    Returns
    -------
    float
        The adjusted value of the parameter at iteration `t` based on the scheduling strategy.
    """
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))
