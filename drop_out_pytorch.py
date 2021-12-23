import torch
import numpy as np
import pandas as pd
import copy
from torch.autograd import grad as gradient
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split


class InfluenceFunction(object):
    def __init__(self, model, X_train, y_train, loss_func, layer_index):
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"Only torch.nn.Module models are supported, got f{type(model)}")
        if not isinstance(X_train, torch.Tensor):
            raise RuntimeError(f"X_train must be <torch.Tensor>, got f{type(X_train)}")
        if not isinstance(y_train, torch.Tensor):
            raise RuntimeError(f"y_train must be <torch.Tensor>, got f{type(y_train)}")

        try:
            layers = model.layers
        except:
            raise RuntimeError('The model must have an attribute "layers".')

        self._model = copy.deepcopy(model)
        self._X_train = X_train
        self._y_train = y_train
        self._loss_func = loss_func
        self._layer_index = layer_index

        self._total_training_loss_grad = None
        self._Hessian_matrix = self._calculate_Hessian_matrix()
        self._Hessian_inv = np.linalg.inv(self._Hessian_matrix)

    def _calculate_Hessian_matrix(self):

        X_train = self._X_train
        y_train = self._y_train
        model = self._model

        y_pred = model(X_train)
        loss = self._loss_func(y_pred, y_train)
        layer = model.layers[self._layer_index]

        if not isinstance(layer, torch.nn.Linear):
            raise RuntimeError(
                f"Only support layer type torch.nn.Linear, got f{type(layer)}."
            )

        weights = model.layers[self._layer_index].weight
        bias = model.layers[self._layer_index].bias

        grad_L_w_1 = gradient(
            loss, (weights, bias), retain_graph=True, create_graph=True
        )
        self._total_training_loss_grad = np.array(
            grad_L_w_1[0][0].tolist() + grad_L_w_1[1].tolist()
        )

        Hessian = []
        for i in range(0, grad_L_w_1[0].shape[1]):
            grad_L_w_2 = gradient(
                grad_L_w_1[0][0][i], (weights, bias), retain_graph=True
            )
            Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())
        grad_L_w_2 = gradient(grad_L_w_1[1][0], (weights, bias), retain_graph=True)
        Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())

        return np.array(Hessian)

    def influence_remove_single(self, index):
        return -self.influence_add_single(self._X_train[index], self._y_train[index])

    def influence_modify_single(self, index, new_x, new_y):
        return self.influence_add_single(new_x, new_y) + self.influence_remove_single(index)

    def influence_add_single(self, x, y):

        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"Added x must be <torch.Tensor>, got f{type(x)}")
        if not isinstance(y, torch.Tensor):
            raise RuntimeError(f"Added y must be <torch.Tensor>, got f{type(y)}")

        x = x.reshape(1, -1).detach()
        y = y.reshape(1, -1).detach()

        model = self._model

        y_pred = model(x)
        loss_single = self._loss_func(y_pred, y)

        weights = model.layers[self._layer_index].weight
        bias = model.layers[self._layer_index].bias

        grad = gradient(loss_single, (weights, bias), retain_graph=True)
        grad = np.array(grad[0][0].tolist() + grad[1].tolist())

        param_offset = - np.dot(self._Hessian_inv, grad)

        influence = np.dot(self._total_training_loss_grad, param_offset)

        return influence

    def get_all_influence(self, X):
        influences1 = []
        for index in range(0, X.shape[0]):
            if index % 100 == 0:
                print("geting {} influence score".format(index))
            influences1.append(self.influence_remove_single(index))
        return influences1

class TorchNNCore(torch.nn.Module):
    def __init__(
        self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU, drop_out=0.2):
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
            elif i == len(struct)-2:
                self.layers.append(hidden_activation())
                self.layers.append(torch.nn.Dropout(drop_out))
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


def run_drop_out(dataset, dpr=0.2, time=0, num_epoch=455):
    torch.set_num_threads(15)
    torch.set_num_interop_threads(15)
    target_file = "Dropout/r={}/influence_{}_{}.npy".format(dpr, dataset, time)
    if os.path.exists(target_file):
        print("Skip {} because it is done".format(target_file))
        return 0
    if not os.path.exists("Dropout/r={}".format(dpr)):
        os.makedirs("Dropout/r={}".format(dpr))
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()

    X_train1 = torch.tensor(data1[:, :-1], dtype=torch.float)
    y_train1 = torch.tensor(data1[:, -1].reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset("One_hot/oh{}.csv".format(dataset))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1 / 80 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 128], hidden_activation=torch.nn.LeakyReLU, drop_out=dpr)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_func = torch.nn.BCELoss()
    for epoch in range(0, num_epoch):
        for X_batch, y_batch in train_loader:
            optim1.zero_grad()
            y_pred1 = model1(X_batch)
            loss1 = loss_func(y_pred1, y_batch)
            loss1.backward()
            optim1.step()
        if epoch % 10 == 0:
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

    np.save(target_file, influences1)


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


class Build_dataset_array(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def gradient_drop_out(dataset, dpr=0.2, time=0, num_epoch=455):
    torch.set_num_threads(15)
    #torch.set_num_interop_threads(15)
    target_file = "Dropout/r={}/acc_{}_{}.npy".format(dpr, dataset, time)
    if os.path.exists(target_file):
        print("Skip {} because it is done".format(target_file))
        return 0
    if not os.path.exists("Dropout/r={}".format(dpr)):
        os.makedirs("Dropout/r={}".format(dpr))
    print("Working on file " + target_file)
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()

    X_train1 = torch.tensor(data1[:, :-1], dtype=torch.float)
    y_train1 = torch.tensor(data1[:, -1].reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset("One_hot/oh{}.csv".format(dataset))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1 / 80 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 128], hidden_activation=torch.nn.LeakyReLU, drop_out=dpr)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_func = torch.nn.BCELoss()
    norm_list = []
    acc_list = []
    for epoch in range(0, num_epoch):
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
    # norm_list = np.vstack(norm_list)
    # np.save(target_file, norm_list)

    acc_list = np.vstack(acc_list)
    np.save(target_file, acc_list)


def run_drop_out_target(dataset, dpr=0.2, time=0, num_epoch=455):
    torch.set_num_threads(15)
    data1 = pd.read_csv("One_hot/oh{}.csv".format(dataset)).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(data1[:, 0:-1], data1[:, -1], test_size=0.5)

    X_train1 = torch.tensor(X_train, dtype=torch.float)
    y_train1 = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float)
    X_test1 = torch.tensor(X_test, dtype=torch.float)
    y_test1 = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float)

    training_dataset = Build_dataset_array(X_train1, y_train1)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=round(1 / 80 * len(X_train1)), shuffle=True)

    model1 = TorchNNCore(inps=X_train1.shape[1], hiddens=[512, 256, 128], hidden_activation=torch.nn.LeakyReLU, drop_out=dpr)
    optim1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    loss_func = torch.nn.BCELoss()
    norm_list = []
    lambda1 = lambda epoch: 0.98 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim1, lr_lambda=lambda1)
    for epoch in range(0, num_epoch):
        for X_batch, y_batch in train_loader:
            optim1.zero_grad()
            y_pred1 = model1(X_batch)
            if epoch % 10 == 0:
                norm_array = check_gradient_norm(model1, X_batch, y_pred1, y_batch, loss_func, epoch)
                norm_list.append(norm_array)
            loss1 = loss_func(y_pred1, y_batch)
            loss1.backward()
            optim1.step()
        if epoch % 10 == 0:
            y_pred_np1 = (y_pred1.detach().numpy()) > 0.5
            accuracy1 = sum(np.array(y_pred_np1) == np.array(y_batch)) / y_batch.shape[0]
            print('Epoch = %d, loss1 = %.4f, accuracy=%.4f' % (epoch, loss1.tolist(), accuracy1))
        if epoch + 1 in [20, 40, 80, 160, 320, 400]:
            target_file = "gradients/Dropout/r={}/{}/epoch={}/target_result.csv".format(dpr, dataset, epoch+1)
            if os.path.exists(target_file):
                print("Skip {} because it is done".format(target_file))
                continue
            if not os.path.exists("gradients/Dropout/r={}/{}/epoch={}".format(dpr, dataset, epoch+1)):
                os.makedirs("gradients/Dropout/r={}/{}/epoch={}".format(dpr, dataset, epoch+1))
            prob_pred_train = model1(X_train1).detach().numpy()
            prob0_pred_train = 1 - prob_pred_train
            label_train = np.array(y_train1)
            g_train = X_train1[:, 0].reshape(-1, 1)
            r_train = X_train1[:, 1].reshape(-1, 1)
            mem_train = np.ones(len(X_train1)).reshape(-1,1)
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
    target_file = "gradients/Dropout/r={}/gradient_Adult.npy".format(dpr, dataset)
    np.save(target_file, norm_list)


if __name__ == "__main__":
    for time in range(1):
        for i in range(1):
            dataset = ['Adult', 'Broward', 'Hospital'][i]
            num_epoch = [400, 1443, 660][i]
            for dpr in [0.01, 0.05, 0.1, 0.2]:
               run_drop_out_target(dataset, dpr, time, num_epoch)