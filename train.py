import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from evaluation import eva, plotClusters
from pytorchtools import EarlyStopping
import time


def scale(z):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std

    return z_scaled


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train(model, epochs, kl_epochs, dataset, views, lr, n_clusters, adj_labels, device, pos_weight, kl_factor,
          tol=1e-8, input_view=0, s_mat=None, beta=0, i=0):
    optimizer = Adam(model.parameters(), lr, weight_decay=1e-4)

    s_mat = torch.Tensor(s_mat).to(device)
    s_mat = F.softmax(s_mat, dim=1)
    features = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    y = np.argmax(y, axis=1)
    views = torch.stack(views, dim=0).to(device)
    adjs = []
    for al in adj_labels:
        adjs.append(torch.Tensor(al))
    adjs = torch.stack(adjs, dim=0).to(device)
    t_pos_weight = torch.Tensor(pos_weight).unsqueeze(-1).to(device)  # todo: shape

    acc_max = nmi_max = ari_max = f1_max = 0.

    model.train()
    es1 = EarlyStopping(50, verbose=True, delta=0.001)

    best = 0.
    tol_time = 0.0

    acc_list = []
    nmi_list = []
    f1_list = []
    ari_list = []
    time_list = []

    for e in tqdm(range(epochs)):
        time.time()
        model.train()
        optimizer.zero_grad()
        adj_loss = 0.
        fea_loss = 0.

        st = time.time()
        embeddings, reconstructed_views, x_bar, _, _, _, _ = model(features, views, input_view, s_mat)
        kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
        acc, nmi, ari, f1 = eva(y, kmeans.labels_, e)
        tqdm.write("Epoch: {}, acc={:.5f}, nmi={:.5f}, ari={:.5f}, f1={:.5f}".format(e + 1, acc, nmi, ari, f1))
        if acc > 0.90:
            break

        # adj matrix reconstruction loss
        for v in range(len(views)):
            adj_loss += F.binary_cross_entropy_with_logits(reconstructed_views[v].reshape([-1]),
                                                           adjs[v].reshape(([-1])), pos_weight=t_pos_weight[v])

        fea_loss = F.mse_loss(x_bar, features)

        # triplet loss
        # zx = embeddings[all_x]
        # zy = embeddings[all_y]
        # pred = model.dcs(zx, zy)
        # sample_loss = F.binary_cross_entropy_with_logits(pred, sample_labels)

        loss = adj_loss + fea_loss
        loss.backward()
        optimizer.step()

        et = time.time()
        tol_time += (et - st)

        # acc_list.append(acc)
        # nmi_list.append(nmi)
        # f1_list.append(f1)
        # ari_list.append(ari)
        # time_list.append(et - st)
    #
    # # print('avg time:', tol_time)

    torch.save(model.state_dict(),'pretrain_acmp')
    exit()
    model.load_state_dict(torch.load('pretrain_acmp'))
    embeddings, reconstructed_views, x_bar, _, _, _, _ = model(features, views, input_view, s_mat)
    kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(embeddings.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, nmi, ari, f1 = eva(y, kmeans.labels_, 0)
    print(acc)
    print(f1)
    print(nmi)

    # embeddings, reconstructed_views, x_bar, _, _, _, _ = model(features, views, input_view, s_mat)
    # for i in range(10):
    #     kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
    #     acc, nmi, ari, f1 = eva(y, kmeans.labels_, 0)
    #     print("Epoch: {}, acc={:.5f}, nmi={:.5f}, ari={:.5f}, f1={:.5f}".format(0, acc, nmi, ari, f1))

    # with open("efficiency_ACM_A_alb_bi.txt", 'w') as f:
    #     for i in range(len(acc_list)):
    #         f.write(str(time_list[i]) + " " + str(acc_list[i]) + " " +
    #                 str(f1_list[i])+" "+str(nmi_list[i])+" "+str(ari_list[i]) + "\n")

    #     if acc>best:
    #         best=acc
    #         bnmi=nmi
    #         bari=ari
    #         bf1=f1
    # print("best:",best,bnmi,bari,bf1)

    # es1(acc, model)
    # if es1.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break
    #
    # if ari > 0.06 and f1 > 0.45:
    #     break

    #     if (e #+ 1) % 10 == 0 or e == 0:
    #     #         kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
    #     #         acc, nmi, ari, f1 = eva(y, kmeans.labels_, e)
    #     #         tqdm.write("Epoch: {}, acc={:.5f}, nmi={:.5f}, ari={:.5f}, f1={:.5f}".format(e + 1, acc, nmi, ari, f1))
    #     #     #######################################################################################################################
    #
    #
    # # kl开始
    # embeddings, reconstructed_views, x_bar, _, _, _, _ = model(features, views, input_view, s_mat)
    # kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(embeddings.data.cpu().numpy())
    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # eva(y, kmeans.labels_, 0)
    # print(acc)
    # print(f1)
    # print(nmi)

    # model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    for params_group in optimizer.param_groups:
        params_group['lr'] = 5e-4
    # #
    # print('start klklklkl')
    # #
    # # es2 = EarlyStopping(10, verbose=True, delta=0.001)
    triplet_loss = nn.TripletMarginLoss(margin=1, p=2)
    # # ################################
    for e in range(kl_epochs):
        model.train()
        optimizer.zero_grad()
        adj_loss = 0.
        fea_loss = 0.
        kl_loss = 0.

        if e % 5 == 0:
            # update interval
            _, _, _, q, _, _, _ = model(features, views, input_view, s_mat)
            p = target_distribution(q.data)

        embeddings, reconstructed_views, x_bar, q, top1, top2, centroids = model(features, views, input_view, s_mat)

        kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
        acc, nmi, ari, f1 = eva(y, kmeans.labels_, e)

        acc_list.append(acc)
        nmi_list.append(nmi)
        f1_list.append(f1)
        ari_list.append(ari)

        tqdm.write("Epoch: {}, acc={:.5f}, nmi={:.5f}, ari={:.5f}, f1={:.5f}".format(e + 1, acc, nmi, ari, f1))

        for v in range(len(views)):
            adj_loss += F.binary_cross_entropy_with_logits(reconstructed_views[v].reshape([-1]),
                                                           adjs[v].reshape(([-1])), pos_weight=t_pos_weight[v])

        centroids_norm = scale(centroids)
        centroids_norm = F.normalize(centroids_norm)
        embeddings_norm = scale(embeddings)
        embeddings_norm = F.normalize(embeddings_norm)
        tc_loss = triplet_loss(embeddings_norm, centroids_norm[top1], centroids[top2])

        fea_loss = F.mse_loss(x_bar, features)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        loss = adj_loss + fea_loss + 10 * kl_loss + 0.01* tc_loss

        tqdm.write(
            "Epoch: {}, training loss={:.5f}, adj loss={:.5f}, fea loss={:.5f}, kl loss={:.5f}".format(e + 1, loss,
                                                                                                       adj_loss,
                                                                                                       fea_loss,
                                                                                                       kl_loss))

        loss.backward()
        optimizer.step()
    ##########################

    # with open("efficiency_BDLP_P_alb_tc_"+str(i)+"_"+str(beta)+"_2.txt", 'w') as f:
    #     for i in range(len(acc_list)):
    #         f.write(str(acc_list[i]) + " " +
    #                 str(f1_list[i]) + " " + str(nmi_list[i]) + " " + str(ari_list[i]) + "\n")

    # es2(acc, model)
    # if es1.early_stop:
    #     print("Early stopping")
    #     # 结束模型训练
    #     break
    #
    ############
    # model.eval()
    # embeddings, _, _, _ ,_,_,_= model(features, views, input_view, s_mat)
    # plotClusters(tqdm, embeddings.data.cpu().numpy(), y)
    ############

    # return acc, nmi, ari, f1
