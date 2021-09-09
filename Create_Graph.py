import os
import pickle
import argparse
import networkx as nx

from tqdm import trange
from torch_geometric.utils import from_networkx
from networkx.generators.random_graphs import powerlaw_cluster_graph

parser = argparse.ArgumentParser()
parser.add_argument('-n_graph', '--n_synthetic_graph', type=int, default=30)
parser.add_argument('-n_node', '--n_node', type=int, default=200)
parser.add_argument('-a', '--average_degree', type=int, default=4)
args = parser.parse_args()

def main(args):
    data_path = f'./data/Synthetic/train/'
    graph_file = f'graph_{args.n_node}_{args.n_synthetic_graph}'
    graph_centrality_file = f'graph_centrality_{args.n_node}_{args.n_synthetic_graph}'

    graph_path = os.path.join(data_path, graph_file)
    graph_centrality_path = os.path.join(data_path, graph_centrality_file)

    graphs = {}
    graph_centralities = {}

    for i in trange(args.n_synthetic_graph):
        nx_graph = powerlaw_cluster_graph(args.n_node, args.average_degree, 0.05)

        graph_centrality = nx.betweenness_centrality(nx_graph)

        graph = from_networkx(nx_graph)

        graphs[i] = graph # Data() type:Object
        graph_centralities[i] = graph_centrality # dict

    with open(graph_path, 'wb') as output:
        pickle.dump(graphs, output, pickle.HIGHEST_PROTOCOL)

    with open(graph_centrality_path, 'wb') as output:
        pickle.dump(graph_centralities, output, pickle.HIGHEST_PROTOCOL)
        
if __name__ == '__main__':
    main(args)