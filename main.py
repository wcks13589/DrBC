import os
import copy
import torch
import pickle
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm, trange
from DrBC import DrBC
from utils import accuracy_top_N, sample_edges
from scipy.stats import kendalltau

from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import degree, to_undirected

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--n_node', type=int, default=200)
parser.add_argument('-s', '--sample_ratio', type=int, default=5)
parser.add_argument('-l', '--lr', type=float, default=1e-5)
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-d', '--embed_dim', type=int, default=128)

def main():
    graph_path = f'./data/Synthetic/train/graph_{args.n_node}_30'
    graph_centrality_path = f'./data/Synthetic/train/graph_centrality_{args.n_node}_30'
    model_weights_path = f'./model_weights/DrBC_{args.n_node}'

    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)

    with open(graph_centrality_path, 'rb') as f:
        graph_centralities = pickle.load(f)


    node_features = {}
    for index, data in graphs.items():
        node_features[index] = torch.ones(args.n_node, 3)
        node_features[index][:,0] = degree(data.edge_index[0])


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DrBC(3, args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    sample_edge_pairs = sample_edges(args.sample_ratio, args.n_node)
    dataset = TensorDataset(sample_edge_pairs)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=args.batch_size)


    test_graph_path = './data/Youtube'

    test_edge_index = pd.read_csv(os.path.join(test_graph_path, 'com-youtube.txt'),
                             delimiter=' ', header=None).to_numpy().T
    test_edge_index = to_undirected(torch.LongTensor(test_edge_index))

    test_node_centrality = pd.read_csv(os.path.join(test_graph_path, 'com-youtube_score.txt'),
                                       delimiter=':', header=None).to_numpy()[:,1]
    test_node_centrality = torch.Tensor(test_node_centrality)

    test_node_features = torch.ones(test_node_centrality.shape[0], 3)
    test_node_features[:,0] = degree(test_edge_index[0], num_nodes=test_node_centrality.shape[0])

    best_kendall = -1

    for epoch in range(args.epochs):

        train_top_k = []
        train_kendall_tau = []
        test_top_k = []
        test_kendall_tau = []

        for i in range(100):
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            x = node_features[i].to(device)
            edge_index = graphs[i].edge_index.to(device)
            y = torch.Tensor(list(graph_centralities[i].values())).view(-1,1).to(device)

            for sample_pair in dataloader:
                model.train()

                sample_pair = sample_pair[0].T.to(device)
                z = model.encode(x, edge_index)
                output = model.decode(z)
                loss = model.loss_f(output, y, sample_pair)

                loss.backward()
                optimizer.step()

                train_top_k.append(accuracy_top_N(output,y))
                train_kendall_tau.append(kendalltau(output.cpu().detach().numpy(),
                                                    y.cpu().detach().numpy())[0])

            test_device = torch.device('cpu')

            model = model.to(test_device)
            model.eval()

            x = test_node_features.to(test_device)
            edge_index = test_edge_index.to(test_device)
            y = test_node_centrality.view(-1,1)

            z = model.encode(x, edge_index)
            output = model.decode(z)

            test_top_k = accuracy_top_N(output,y)

            test_kendall_tau = kendalltau(output.cpu().detach().numpy(),
                                     y.cpu().detach().numpy())[0]

            print(f'epoch:{epoch+1} Train Top-K: {np.mean(train_top_k)*100},Test Top-K: {test_top_k*100}\n\
            Train Kendall tau: {np.mean(train_kendall_tau)}, Test Kendall tau: {test_kendall_tau}')

            if test_kendall_tau > best_kendall:
                best_model_params = copy.deepcopy(model.state_dict())
                best_kendall = test_kendall_tau
                print('\tBetter!')

    model.load_state_dict(best_model_params)
    model.eval()
    torch.save(model.state_dict(),model_weights_path)

    model = model.to(test_device)
    model.eval()

    x = test_node_features.to(test_device)
    edge_index = test_edge_index.to(test_device)
    y = test_node_centrality.view(-1,1)

    z = model.encode(x, edge_index)
    output = model.decode(z)

    test_top_k = accuracy_top_N(output,y)
    test_kendall_tau = kendalltau(output.cpu().detach().numpy(),
                             y.cpu().detach().numpy())[0]

    return test_top_k, test_kendall_tau

if __name__ == '__main__':
    test_top_k, test_kendall_tau = main(args)
    
    print(f'Final Test top_N:{test_top_k} \
    \n Final Test Kendall_tau:{test_kendall_tau}')
