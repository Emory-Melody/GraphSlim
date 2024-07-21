import torch
import streamlit as st
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GraphSlim Visualization",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Interactive Visualization of GraphSlim')

st.sidebar.image("logo.png")
# st.sidebar.markdown("GraphSlim")
st.write(
    "<style> #input-container { position: fixed; bottom: 0; width: 100%; padding: 10px; background-color: white; z-index: 100; } h1, h2 { font-weight: bold; background: -webkit-linear-gradient(left, red, orange); background: linear-gradient(to right, red, orange); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline; font-size: 3em; } .user-avatar { float: right; width: 40px; height: 40px; margin-left: 5px; margin-bottom: -10px; border-radius: 50%; object-fit: cover; } .bot-avatar { float: left; width: 40px; height: 40px; margin-right: 5px; border-radius: 50%; object-fit: cover; } </style>",
    unsafe_allow_html=True)
method = st.sidebar.selectbox(
    'Method',
    ('random', 'kcenter', 'gcond', 'gcondx')
)

dataset = st.sidebar.selectbox(
    'Dataset',
    ('cora',)
)

reduction_rate = st.sidebar.selectbox(
    'Reduction Rate',
    ('0.1', '0.25', '0.5')
)

st.markdown('### Original Graph')
dataset = Planetoid(root='', name='Cora')
data = dataset[0]

G = to_networkx(data, to_undirected=True)
node_labels = {i: int(data.y[i]) for i in range(data.num_nodes)}
unique_labels = list(set(node_labels.values()))
colors = plt.cm.get_cmap('jet', len(unique_labels))
color_mapping = {label: colors(i) for i, label in enumerate(unique_labels)}
node_colors = {i: mcolors.to_hex(color_mapping[node_labels[i]]) for i in G.nodes}
nx.set_node_attributes(G, node_colors, "color")

net = Network(
    height='400px',
    width='100%'
)

net.from_nx(G)
net.repulsion()

# net.repulsion(
#             node_distance=420,
#             central_gravity=0.33,
#             spring_length=110,
#             spring_strength=0.10,
#             damping=0.95
#             )

# try:
#     path = 'tmp'
#     net.save_graph(f'{path}/pyvis_graph.html')
#     HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

#     # Save and read graph as HTML file (locally)
# except:
path = 'html_files'
net.save_graph(f'{path}/pyvis_graph.html')
HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
# Load HTML file in HTML component for display on Streamlit page

components1 = components.html(HtmlFile.read(), height=500)

st.markdown('### Reduced Graph')
adj_path = f'reduced_graph/{method}/adj_cora_{reduction_rate}_1.pt'
label_path = f'reduced_graph/{method}/label_cora_{reduction_rate}_1.pt'

adj_matrix = torch.load(adj_path, map_location=torch.device('cpu')).to_dense().numpy()
node_labels = torch.load(label_path, map_location=torch.device('cpu')).numpy()

G = nx.from_numpy_array(adj_matrix)
node_labels_dict = {i: int(node_labels[i]) for i in range(len(node_labels))}
unique_labels = list(set(node_labels_dict.values()))
colors = plt.cm.get_cmap('jet', len(unique_labels))
color_mapping = {label: colors(i) for i, label in enumerate(unique_labels)}
node_colors = {i: mcolors.to_hex(color_mapping[node_labels[i]]) for i in G.nodes}
nx.set_node_attributes(G, node_colors, 'color')

net = Network(
    height='400px',
    width='100%'
)

net.from_nx(G)
net.repulsion()
# try:
#     path = '/tmp'
#     net.save_graph(f'{path}/reduced_graph.html')
#     HtmlFile = open(f'{path}/reduced_graph.html', 'r', encoding='utf-8')
#     # Save and read graph as HTML file (locally)
# except:
path = 'html_files'
net.save_graph(f'{path}/reduced_graph.html')
HtmlFile = open(f'{path}/reduced_graph.html', 'r', encoding='utf-8')
# Load HTML file in HTML component for display on Streamlit page

components2 = components.html(HtmlFile.read(), height=500)

st.balloons()
