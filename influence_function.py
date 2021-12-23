import torch
import numpy as np

import copy
from torch.autograd import grad as gradient
import sys


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
        print("Heessian invers is finished")

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
            y_pred = model(X_train)
            sys.stdout.write("\rWorking on Hessian point %i" % i)
            sys.stdout.flush()
            grad_L_w_2 = gradient(
                grad_L_w_1[0][0][i], (weights, bias), retain_graph=True, create_graph=True
            )
            Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())
        y_pred = model(X_train)
    
        Hessian.append(grad_L_w_2[0][0].tolist() + grad_L_w_2[1].tolist())
        print("Hessian is finished")

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
            sys.stdout.write("\rWorking on Hessian influence %i" % index)
            sys.stdout.flush()
            influences1.append(self.influence_remove_single(index))
        return influences1