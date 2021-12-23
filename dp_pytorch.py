import torch
import numpy as np
import pandas as pd
from opacus import PrivacyEngine
from torch.utils.data import Dataset
from influence_function import InfluenceFunction
from torch.autograd import grad as gradient
import os
from sklearn.model_selection import train_test_split


class TorchNNCore(torch.nn.Module):
    def __init__(
        self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU
    ):
        super(TorchNNCore, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        struct = [inps] + hiddens + [1]
        self.layers = [] # This layer attribute is required under
        for i in range(1, len(struct)):
            self.layers.append(
                torch.nn.Linear(
                    in_features=struct[i - 1], out_features=struct[i], bias=bias
                )
            )
            if i == len(struct) - 1:
                self.layers.append(torch.nn.Sigmoid())
            else:
                self.layers.append(hidden_activation())
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.model(x)
        return output


class Build_dataset(Dataset):
    def __init__(self, file):
        data = pd.read_csv(file).to_numpy()
        self.X = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.y = torch.tensor(data[:, -1].reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def check_gradient_norm(model, X_batch, y_pred, y_label, loss_func, epoch):
    norm_list = []
    s = X_batch.numpy()[:, 0]
    y = y_label.numpy().reshape(-1)
    for yi in [0, 1]:
        for si in [0, 1]:
            ind = (y == yi) * (s == si)
            x_tmp = X_batch[ind]
            y_pred_i = model(x_tmp)
            y_label_i = y_label[ind]
            loss_i = loss_func(y_pred_i, y_label_i)
            grad = gradient(loss_i, model.parameters(), retain_graph=True)
            total_norm = 0
            for g_layer in grad:
                param_norm = g_layer.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            norm_list.append(total_norm)
    # get epoch, s and y
    ep = np.ones([4]) * epoch
    s = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    norm_array = np.array([y, s, ep, norm_list]).T
    return norm_array


def check_acc_sub(model, X_batch, y_pred, y_label, epoch):
    acc_list = []
    s = X_batch.numpy()[:, 0]
    y = y_label.numpy().reshape(-1)
    for yi in [0, 1]:
        for si in [0, 1]:
            ind = (y == yi) * (s == si)
            x_tmp = X_batch[ind]
            y_pred_i = model(x_tmp)
            y_label_i = y_label[ind]
            y_pred_np1 = (y_pred_i.detach().numpy()) > 0.5
            accuracy1 = sum(np.array(y_pred_np1) == np.array(y_label_i)) / len(y_label_i)
            acc_list.append(float(accuracy1))
    # get epoch, s and y
    ep = np.ones([4]) * epoch
    s = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    norm_array = np.array([y, s, ep, acc_list]).T
    return norm_array


def run_DP_influence(dataset, time, ep, epoch):
    target_file = "DP/ep={}/influence_{}_{}.npy".format(ep, dataset, time)
    if os.path.exists(target_file):
        print("Skip {} because it is done".format(target_file))
        return 0
    if not os.path.exists("DP/ep={}".format(ep)):
        os.mkdir("DP/ep={}".format(ep))
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()

    X_train1 = torch.tensor(data1[:, :-1], dtype=torch.float)
    y_train1 = torch.tensor(data1[:, -1].reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset("One_hot/oh{}.csv".format(dataset))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1/100 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 96], hidden_activation=torch.nn.LeakyReLU)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    privacy_engine = PrivacyEngine(
        model1,
        epochs=epoch,
        sample_rate=1/100,
        target_epsilon=ep,
        target_delta=1e-6,
        max_grad_norm=10,
    )
    privacy_engine.attach(optim1)
    loss_func = torch.nn.BCELoss()
    for epoch in range(0, epoch):
        for X_batch, y_batch in train_loader:
            optim1.zero_grad()
            y_pred1 = model1(X_batch)
            loss1 = loss_func(y_pred1, y_batch)
            loss1.backward()
            optim1.step()
        if epoch % 1 == 0:
            y_pred_np1 = (y_pred1.detach().numpy()) > 0.5
            accuracy1 = sum(np.array(y_pred_np1) == np.array(y_batch)) / y_batch.shape[0]
            print('Epoch = %d, loss1 = %.4f, accuracy=%.4f' % (epoch, loss1.tolist(), accuracy1))
    print('\n')
    optim1.zero_grad()

    infl1 = InfluenceFunction(
        model=model1,
        X_train=X_train1,
        y_train=y_train1,
        loss_func=loss_func,
        layer_index=-2
    )

    influences1 = infl1.get_all_influence(X_train1)

    np.save("DP/ep={}/influence_{}_{}.npy".format(ep, dataset, time), influences1)


def gradient_DP(dataset, time, ep, epoch):
    target_file = "DP/ep={}/acc_{}_{}.npy".format(ep, dataset, time)
    if os.path.exists(target_file):
        pass
        print("Skip {} because it is done".format(target_file))
        return 0
    if not os.path.exists("DP/ep={}".format(ep)):
        os.mkdir("DP/ep={}".format(ep))
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()

    X_train1 = torch.tensor(data1[:, :-1], dtype=torch.float)
    y_train1 = torch.tensor(data1[:, -1].reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset("One_hot/oh{}.csv".format(dataset))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1/100 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 96], hidden_activation=torch.nn.LeakyReLU)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    privacy_engine = PrivacyEngine(
        model1,
        epochs=epoch,
        sample_rate=1/100,
        target_epsilon=ep,
        target_delta=1e-6,
        max_grad_norm=10,
    )
    privacy_engine.attach(optim1)
    loss_func = torch.nn.BCELoss()
    norm_list = []
    acc_list = []
    for epoch in range(0, epoch):
        for X_batch, y_batch in train_loader:
            optim1.zero_grad()
            y_pred1 = model1(X_batch)
            if epoch % 5 == 0:
                pass
                #norm_array = check_gradient_norm(model1, X_batch, y_pred1, y_batch, loss_func, epoch)
                #norm_list.append(norm_array)
            loss1 = loss_func(y_pred1, y_batch)
            loss1.backward()
            optim1.step()
        if epoch % 5 == 0:
            y_pred_all = model1(X_train1).detach().numpy() > 0.5
            acc_array = check_acc_sub(model1, X_train1, y_pred_all, y_train1, epoch)
            acc_list.append(acc_array)
            accuracy1 = sum(np.array(y_pred_all) == np.array(y_train1)) / len(y_train1)
            print('Epoch = %d, loss1 = %.4f, accuracy=%.4f' % (epoch, loss1.tolist(), accuracy1))
    print('\n')
    optim1.zero_grad()

    #norm_list = np.vstack(norm_list)
    #np.save(target_file, norm_list)

    acc_list = np.vstack(acc_list)
    np.save(target_file, acc_list)


class Build_dataset_array(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def run_DP_target(dataset, time, ep, epoch):
    torch.set_num_threads(15)
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(data1[:, 0:-1], data1[:, -1], test_size=0.5)

    X_train1 = torch.tensor(X_train, dtype=torch.float)
    y_train1 = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test1 = torch.tensor(X_test, dtype=torch.float)
    y_test1 = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset_array(X_train1, y_train1)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1 / 80 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 96], hidden_activation=torch.nn.LeakyReLU)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    lambda1 = lambda epoch: 0.98 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim1, lr_lambda=lambda1)
    privacy_engine = PrivacyEngine(
        model1,
        epochs=epoch,
        sample_rate=1/100,
        target_epsilon=ep,
        target_delta=1e-6,
        max_grad_norm=10,
    )
    privacy_engine.attach(optim1)
    loss_func = torch.nn.BCELoss()
    norm_list = []
    for epoch in range(0, epoch):
        for X_batch, y_batch in train_loader:
            optim1.zero_grad()
            y_pred1 = model1(X_batch)
            if epoch % 10 == 0:
                norm_array = check_gradient_norm(model1, X_batch, y_pred1, y_batch, loss_func, epoch)
                norm_list.append(norm_array)
            loss1 = loss_func(y_pred1, y_batch)
            loss1.backward()
            optim1.step()
        if epoch % 5 == 0:
            y_pred_np1 = (y_pred1.detach().numpy()) > 0.5
            accuracy1 = sum(np.array(y_pred_np1) == np.array(y_batch)) / y_batch.shape[0]
            print('Epoch = %d, loss1 = %.4f, accuracy=%.4f' % (epoch, loss1.tolist(), accuracy1))
        if epoch + 1 in [20, 40, 80, 160, 320, 400]:
            target_file = "gradients/DP/ep={}/{}/epoch={}/target_result.csv".format(ep, dataset, epoch+1)
            if os.path.exists(target_file):
                print("Skip {} because it is done".format(target_file))
                continue
            if not os.path.exists("gradients/DP/ep={}/{}/epoch={}".format(ep, dataset, epoch+1)):
                os.makedirs("gradients/DP/ep={}/{}/epoch={}".format(ep, dataset, epoch+1))
            prob_pred_train = model1(X_train1).detach().numpy()
            prob0_pred_train = 1 - prob_pred_train
            label_train = np.array(y_train1)
            g_train = X_train1[:, 0].reshape(-1, 1)
            r_train = X_train1[:, 1].reshape(-1, 1)
            mem_train = np.ones(len(X_train1)).reshape(-1, 1)
            all_train = np.hstack([label_train, prob0_pred_train, prob_pred_train, g_train, r_train, mem_train])

            prob_pred_test = model1(X_test1).detach().numpy()
            prob0_pred_test = 1 - prob_pred_test
            label_test = np.array(y_test1)
            g_test = X_test1[:, 0].reshape(-1, 1)
            r_test = X_test1[:, 1].reshape(-1, 1)
            mem_test = np.zeros(len(X_test1)).reshape(-1, 1)
            all_test = np.hstack([label_test, prob0_pred_test, prob_pred_test, g_test, r_test, mem_test])

            df_all = pd.DataFrame(np.vstack([all_train, all_test]))
            df_all.to_csv(target_file, header=False, index=False)
        scheduler.step()
    norm_list = np.vstack(norm_list)
    target_file = "gradients/DP/ep={}/gradient_{}.npy".format(ep, dataset)
    np.save(target_file, norm_list)



if __name__ == "__main__":
    settings = {"Adult": [[0.3, 5], [0.5, 10], [1.0, 30], [3.0, 100], [5.0, 200]],
                "Broward": [[0.3, 12], [0.5, 23], [1.0, 91], [3.0, 300], [5.0, 600]],
                "Hospital": [[0.3, 10], [0.5, 15], [1.0, 25], [3.0, 40], [5.0, 100]]}
    torch.set_num_threads(15)
    for time in range(1):
        for d in ['Adult']:
            for i in range(0, 4):
                ep = settings[d][i][0]
                epoch = 400
                run_DP_target(d, time, ep, epoch)