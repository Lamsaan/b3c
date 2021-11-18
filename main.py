from __future__ import print_function, division
import torch
import argparse
from sklearn.decomposition import PCA
from constructor import format_data
from utils import *
from models import *
from train import train
import scipy.io as sio
from tqdm import tqdm
from evaluation import eva, plotClusters


class load_data(Dataset):
    def __init__(self, features, labels):
        self.x = features  # features
        self.y = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))


def to_torch_sparse_tensor(matrices):
    ret = []
    for mat in matrices:
        ret.append(sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(mat)))
    return ret


# def train(model,epoch,features,views,label,lr,n_clusters):
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # def train(model, epochs, kl_epochs, dataset, views, lr, n_clusters, adj_labels, device, pos_weight, kl_factor,
    #           tol=0.001):
    # (self, in_features, hidden, out_features, n_view, order, n_cluster,
    #  dropout=0.)
    parser.add_argument('--dataset', type=str, default='ACM')
    parser.add_argument('--order', nargs='+', type=int, default=[0, 0])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--kl_epochs', type=int, default=500)
    parser.add_argument('--kl_factor', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--hidden', nargs='+', type=int, default=[32,32])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input_view',type=int,default=0)
    parser.add_argument('--ntype',type=str)


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('use cuda:{}'.format(args.cuda))
    device = torch.device('cuda' if args.cuda else 'cpu')

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    # # feas = {'adjs': adjs_norm, 'adjs_label': adjs_label, 'num_features': num_features, 'num_nodes': num_nodes,
    # #         'true_labels': true_labels, 'pos_weights': pos_weights, 'norms': np.array(norms), 'adjs_norm': adjs_norm,
    # #         'features': features, 'fea_pos_weights': fea_pos_weights, 'numView': numView}

    if args.dataset == 'ACM':
        input_view = 0
    if args.dataset == 'DBLP':
        input_view = 1
    if args.dataset == 'IMDB':
        input_view = 1
    input_view=args.input_view
    all_data = format_data(args.dataset,args.ntype)
    views = to_torch_sparse_tensor(all_data['adjs_norm'])
    # weighted_view=to_torch_sparse_tensor(all_data['weighted_adj_norm'])

    s_mat = np.matmul(all_data['features'], all_data['features'].T)
    #
    pca = PCA(n_components=100)
    all_data['features'] = pca.fit_transform(all_data['features'])
    # plotClusters(tqdm, all_data['features'], np.argmax(all_data['true_labels'], axis=1))

    dataset = load_data(all_data['features'], all_data['true_labels'])
    model = B3C(in_features=dataset.x.shape[1], hidden=args.hidden, out_features=args.out,
                n_views=all_data['numView'], n_cluster=args.n_clusters, dropout=args.dropout).to(device)

    train(model=model, epochs=args.epochs, kl_epochs=args.kl_epochs, dataset=dataset, views=views,
          lr=args.lr, n_clusters=args.n_clusters, adj_labels=all_data['adjs_label'],
          device=device, pos_weight=all_data['pos_weights'], kl_factor=args.kl_factor, tol=1e-8, input_view=input_view,
          s_mat=s_mat)

    # for beta in [0.1,1.0,10.0]:
    #     for i in range(5):
    #         print("running paras:",beta,i)
    #         model = B3C(in_features=dataset.x.shape[1], hidden=args.hidden, out_features=args.out,
    #                          n_views=all_data['numView'], order=args.order, n_cluster=args.n_clusters, dropout=args.dropout).to(
    #                     device)
    #
    #         train(model=model, epochs=args.epochs, kl_epochs=args.kl_epochs, dataset=dataset, views=views,
    #                 lr=args.lr, n_clusters=args.n_clusters, adj_labels=all_data['adjs_label'], samples=all_data['samp'],
    #                 device=device, pos_weight=all_data['pos_weights'], kl_factor=args.kl_factor, tol=1e-8, input_view=input_view,
    #                 batch_size=args.batch_size, s_mat=s_mat,beta=beta,i=i)


