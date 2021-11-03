import tensorflow.compat.v2 as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from keras import backend as K
from keras.layers import Dense, Activation, Input
from keras.models import Model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from sklearn import metrics
import os


class target_model():
    def __init__(self,
                 num_epoch=60,
                 dp_flag=0,
                 l2_norm_clip=1.0,
                 noise_multiplier=1.3,
                 num_microbatches=25,
                 learning_rate=0.01,
                 data_size=10000):
        self.nm = noise_multiplier
        self.l2 = l2_norm_clip
        self.t_model = tf.keras.Sequential([
            # tf.keras.Input(shape=(input_size,)),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.num_epoch = num_epoch
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=3000,
            decay_rate=0.96,
            staircase=True)
        # self.lr_schedule = learning_rate
        self.batch_size = num_microbatches
        self.data_size = data_size
        self.dp = dp_flag
        if dp_flag:
            self.set_opt()
        else:
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
            # self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_fn = keras.losses.categorical_crossentropy(from_logits=True)
        self.t_model.compile(optimizer=self.opt,
                             loss=self.loss_fn,
                             # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                             metrics=['accuracy'])

    def fit(self, X_train, y_train):
        if self.dp:
            number_of_batch = self.data_size // self.batch_size
            inds = np.arange(X_train.shape[0])
            for epoch in range(self.num_epoch):
                # Train the model for one epoch.
                np.random.shuffle(inds)
                # for (_, (images, labels)) in enumerate(dataset.take(-1)):
                for batch_ind in range(number_of_batch):
                    X_batch = X_train[inds[(batch_ind) * self.batch_size:min((batch_ind + 1) * self.batch_size,
                                                                             X_train.shape[0])], :]
                    y_batch = y_train[
                        inds[(batch_ind) * self.batch_size:min((batch_ind + 1) * self.batch_size, X_train.shape[0])]]
                    with tf.GradientTape(persistent=True) as gradient_tape:
                        # This dummy call is needed to obtain the var list.
                        logits = self.t_model(X_batch, training=True)
                        var_list = self.t_model.trainable_variables

                        def loss_fn():
                            logits = self.t_model(X_batch,
                                                  training=True)  # pylint: disable=undefined-loop-variable,cell-var-from-loop
                            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_batch.astype('int'),
                                                                                  logits=logits)
                            return loss

                        grads_and_vars = self.opt.compute_gradients(loss_fn, var_list, gradient_tape=gradient_tape)

                    self.opt.apply_gradients(grads_and_vars)
                    y_pred = self.t_model.predict(X_train)
                    y_pred = np.argmax(y_pred, axis=1)
                    acc = sum(y_pred == y_train) / len(y_pred)

                    print('Test accuracy after batch %d is: %.3f' % (batch_ind, acc), end='\r')
                    sys.stdout.flush()

                # Evaluate the model and print results
                '''y_pred = self.t_model.predict(X_train)
                y_pred = np.argmax(y_pred, axis=1)
                acc = sum(y_pred == y_train) / len(y_pred)

                print('Test accuracy after epoch %d is: %.3f' % (epoch, acc))'''
        else:
            self.t_model.fit(X_train, y_train, epochs=self.num_epoch, verbose=2, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.t_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.t_model.predict(X)
        y_pred = probs.argmax(axis=1)
        return y_pred

    def score(self, X_val, y_val):
        probs = self.t_model.predict(X_val)
        y_pred = probs.argmax(axis=1)
        acc = metrics.accuracy_score(y_val, y_pred)
        return acc

    def set_opt(self):
        self.opt = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=self.l2,
            noise_multiplier=self.nm,
            num_microbatches=self.batch_size,
            learning_rate=self.lr_schedule)
        '''self.ep = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=self.data_size,
                                                                batch_size=self.batch_size,
                                                                noise_multiplier=self.nm,
                                                                epochs=self.num_epoch,
                                                                delta=1e-5)[0]'''

    def clf_prob(self, X):
        plables = self.predict(X)
        prob = self.predict_proba(X)
        return prob, plables


class target_model2():
    def __init__(self,
                 num_epoch=60,
                 dp_flag=0,
                 l2_norm_clip=1.0,
                 noise_multiplier=1.3,
                 num_microbatches=25,
                 learning_rate=0.01,
                 data_size=10000,
                 verbos=1):
        self.nm = noise_multiplier
        self.l2 = l2_norm_clip
        self.dp = dp_flag
        self.num_epoch = num_epoch
        self.batch_size = num_microbatches
        self.data_size = data_size
        self.verbos = verbos
        if self.dp:
            layers = [tf.keras.layers.Dense(512, activation='relu'),
                      tf.keras.layers.Dense(256, activation='relu'),
                      tf.keras.layers.Dense(128, activation='relu'),
                      tf.keras.layers.Dense(2, activation='softmax')]
            self.t_model = Sequential(
                l2_norm_clip=self.l2,
                noise_multiplier=self.nm,
                layers=layers)

        else:
            self.t_model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True)
        # self.lr_schedule = learning_rate
        if self.dp:
            '''self.opt = DPKerasSGDOptimizer(
                l2_norm_clip=self.l2,
                noise_multiplier=self.nm,
                num_microbatches=self.batch_size,
                learning_rate=self.lr_schedule)'''
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction=tf.keras.losses.Reduction.NONE)
        else:
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                      reduction=tf.keras.losses.Reduction.NONE)
        self.t_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X_train, y_train):
        y_train = y_train.reshape(-1, 1)
        # y_train = np.hstack([1 - y_train, y_train])
        self.t_model.fit(X_train, y_train, epochs=self.num_epoch, verbose=self.verbos, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.t_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.t_model.predict(X)
        y_pred = probs.argmax(axis=1)
        return y_pred

    def clf_prob(self, X):
        plables = self.predict(X)
        prob = self.predict_proba(X)
        return prob, plables


class target_model3():
    def __init__(self,
                 num_epoch=60,
                 dp_flag=0,
                 l2_norm_clip=1.0,
                 noise_multiplier=1.3,
                 num_microbatches=25,
                 learning_rate=0.01,
                 data_size=10000,
                 verbos=1,
                 reduce=1):
        self.nm = noise_multiplier
        self.l2 = l2_norm_clip
        self.dp = dp_flag
        self.num_epoch = num_epoch
        self.batch_size = num_microbatches
        self.data_size = data_size
        self.verbos = verbos
        self.t_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')])
        if reduce:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True)
        else:
            self.lr_schedule = learning_rate
        if self.dp:
            self.opt = DPKerasSGDOptimizer(
                l2_norm_clip=self.l2,
                noise_multiplier=self.nm,
                num_microbatches=self.batch_size,
                learning_rate=self.lr_schedule)
        else:
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.t_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X_train, y_train, epoch=-1):
        if epoch == -1:
            epoch = self.num_epoch
        y_train = y_train.reshape(-1, 1)
        self.t_model.fit(X_train, y_train, epochs=epoch, verbose=self.verbos, batch_size=self.batch_size)

    def fit_ds(self, dataset, epoch=-1):
        if epoch == -1:
            epoch = self.num_epoch
        self.t_model.fit(dataset, epochs=epoch, verbose=self.verbos, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.t_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.t_model.predict(X)
        y_pred = probs.argmax(axis=1)
        return y_pred

    def score(self, X_val, y_val):
        probs = self.t_model.predict(X_val)
        y_pred = probs.argmax(axis=1)
        acc = metrics.accuracy_score(y_val, y_pred)
        return acc

    def clf_prob(self, X):
        plables = self.predict(X)
        prob = self.predict_proba(X)
        return prob, plables

    def save_weight(self, location):
        if not os.path.exists(location):
            os.mkdir(location)
        self.t_model.save_weights(location+"/tmp_model")


class target_model_E():
    def __init__(self,
                 num_epoch=60,
                 dp_flag=0,
                 l2_norm_clip=1.0,
                 noise_multiplier=1.3,
                 num_microbatches=25,
                 learning_rate=0.01,
                 data_size=10000,
                 verbos=1,
                 reduce=1):
        self.nm = noise_multiplier
        self.l2 = l2_norm_clip
        self.dp = dp_flag
        self.num_epoch = num_epoch
        self.batch_size = num_microbatches
        self.data_size = data_size
        self.verbos = verbos
        self.female_gradient_record = []
        self.male_gradient_record = []
        self.pr_gradient_record = []
        self.unpr_gradient_record = []
        self.t_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')])
        if reduce:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True)
        else:
            self.lr_schedule = learning_rate
        if self.dp:
            self.opt = DPKerasSGDOptimizer(
                l2_norm_clip=self.l2,
                noise_multiplier=self.nm,
                num_microbatches=self.batch_size,
                learning_rate=self.lr_schedule)
        else:
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.NONE)
        self.t_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X, y, Xt, yt, epoch=-1):
        # Training loop.
        steps_per_epoch = len(X) / self.batch_size
        if epoch == -1:
            n_epoch = self.num_epoch
        else:
            n_epoch = epoch
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(1000).batch(self.batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((Xt, yt))
        test_dataset = test_dataset.batch(self.batch_size)
        for epoch in range(n_epoch):
            if epoch >= 800:
                pass
                # break
            if epoch % 100 == 0:
                print("output a figure of gender gradient record")
                plt.figure("Gradient Record")
                plt.plot(self.female_gradient_record)
                plt.plot(self.male_gradient_record)
                plt.xlabel("Step")
                plt.ylabel("Gradient (Average of L2 norm)")
                plt.legend(["Female", "Male"])
                plt.savefig("Da_result/Gender_gradient_record_epoch=" + str(epoch) + ".png")
                plt.close()

                print("output a figure of race gradient record")
                plt.figure("Gradient Record")
                plt.plot(self.pr_gradient_record)
                plt.plot(self.unpr_gradient_record)
                plt.xlabel("Step")
                plt.ylabel("Gradient (Average of L2 norm)")
                plt.legend(["Protected Race", "Unprotected Race"])
                plt.savefig("Da_result/Race_gradient_record_epoch=" + str(epoch) + ".png")
                plt.close()

            # Train the model for one epoch.
            step = 0
            for (_, (Xs, ys)) in enumerate(dataset.take(-1)):
                print("Starting step", step, "out of", round(steps_per_epoch) - 2, end='\r')
                sys.stdout.flush()
                step = step + 1
                male_ind = Xs[:, 0] == 1
                female_ind = Xs[:, 0] == 0
                pr_ind = Xs[:, 1] == 1
                unpr_ind = Xs[:, 1] == 0
                with tf.GradientTape(persistent=True) as gradient_tape:
                    # This dummy call is needed to  the var list.
                    logits = self.t_model(Xs, training=True)
                    var_list = self.t_model.trainable_variables

                    # In Eager mode, the optimizer takes a function that returns the loss.
                    def loss_fn():
                        logits = self.t_model(Xs,
                                              training=True)  # pylint: disable=undefined-loop-variable,cell-var-from-loop
                        loss = tf.keras.losses.sparse_categorical_crossentropy(ys, logits, from_logits=True, axis=1)
                        # If training without privacy, the loss is a scalar not a vector.
                        if not self.dp:
                            loss = tf.reduce_mean(input_tensor=loss)
                        return loss

                if self.dp:
                    grads_and_vars = self.compute_gradients(loss_fn, var_list, tape=gradient_tape,
                                                            male_ind=male_ind, female_ind=female_ind,
                                                            protect_race_ind=pr_ind, unprotect_race_ind=unpr_ind)
                else:
                    grads_and_vars = self.compute_gradients(loss_fn, var_list, male_ind=male_ind, female_ind=female_ind)

                self.opt.apply_gradients(grads_and_vars)

            # Evaluate the model and print results
            for (_, (Xt, yt)) in enumerate(test_dataset.take(-1)):
                logits = self.t_model(Xt, training=False)
                correct_preds = tf.equal(tf.argmax(input=logits, axis=1), tf.cast(yt, tf.int64))
            test_accuracy = np.mean(correct_preds.numpy())
            print('Test accuracy after epoch %d is: %.3f' % (epoch, test_accuracy))
            if test_accuracy >= 0.40 and epoch == 0:
                print("First epoch too good! Retry!")
                # return True
            elif epoch == 0:
                print("Pass the first epoch test!")
        return False

    def compute_gradients(self, loss,
                          var_list,
                          grad_loss=None,
                          tape=None,
                          male_ind=[],
                          female_ind=[],
                          protect_race_ind=[],
                          unprotect_race_ind=[]):
        """DP version of superclass method."""

        # Compute loss.
        if not callable(loss) and tape is None:
            raise ValueError('`tape` is required when a `Tensor` loss is passed.')
        tape = tape if tape is not None else tf.GradientTape()

        if callable(loss):
            with tape:
                if not callable(var_list):
                    tape.watch(var_list)

                if callable(loss):
                    loss = loss()
                    microbatch_losses = tf.reduce_mean(
                        tf.reshape(loss, [self.batch_size, -1]), axis=1)

                if callable(var_list):
                    var_list = var_list()
        else:
            with tape:
                microbatch_losses = tf.reduce_mean(
                    tf.reshape(loss, [self.batch_size, -1]), axis=1)

        var_list = tf.nest.flatten(var_list)

        # Compute the per-microbatch losses using helpful jacobian method.
        with tf.keras.backend.name_scope('SGD/gradients'):
            jacobian = tape.jacobian(microbatch_losses, var_list)

            # Clip gradients to given l2_norm_clip.
            def clip_gradients(g):
                return tf.clip_by_global_norm(g, self.l2)[0]

            clipped_gradients = tf.map_fn(clip_gradients, jacobian)
            male_gradient = 0
            female_gradient = 0
            protect_race_gradient = 0
            unprotect_race__gradient = 0
            for i in range(len(clipped_gradients)):
                if len(clipped_gradients[i].shape) == 3:
                    male_gradient += tf.reduce_mean(tf.norm(clipped_gradients[i][male_ind], ord=2, axis=(1, 2))).numpy()
                    female_gradient += tf.reduce_mean(
                        tf.norm(clipped_gradients[i][female_ind], ord=2, axis=(1, 2))).numpy()
                    protect_race_gradient += tf.reduce_mean(
                        tf.norm(clipped_gradients[i][protect_race_ind], ord=2, axis=(1, 2))).numpy()
                    unprotect_race__gradient += tf.reduce_mean(
                        tf.norm(clipped_gradients[i][unprotect_race_ind], ord=2, axis=(1, 2))).numpy()
            self.male_gradient_record.append(male_gradient)
            self.female_gradient_record.append(female_gradient)
            self.pr_gradient_record.append(protect_race_gradient)
            self.unpr_gradient_record.append(unprotect_race__gradient)

            def reduce_noise_normalize_batch(g):
                # Sum gradients over all microbatches.
                summed_gradient = tf.reduce_sum(g, axis=0)

                # Add noise to summed gradients.
                noise_stddev = self.l2 * self.nm
                noise = tf.random.normal(
                    tf.shape(input=summed_gradient), stddev=noise_stddev)
                noised_gradient = tf.add(summed_gradient, noise)

                # Normalize by number of microbatches and return.
                return tf.truediv(noised_gradient, self.batch_size)

            final_gradients = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                    clipped_gradients)

        return list(zip(final_gradients, var_list))


def KerasNeuralModel(struct, lr, output_activation='softmax'):
    sequential = [
        Input(shape=(struct[0][0],))
    ]
    for i in range(1, len(struct)):
        sequential.append(
            Dense(struct[i][0], activation='relu')(sequential[-1])
        )
    outputs_logits = Dense(struct[-1][1])(sequential[-1])
    outputs = Activation(output_activation)(outputs_logits)
    model = Model(inputs=sequential[0], outputs=outputs)
    if output_activation == 'softmax':
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=lr),
                      metrics=['accuracy'])
    elif output_activation == 'sigmoid':
        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(lr=lr),
                      metrics=['accuracy'])
    else:
        raise ValueError(f'Unsupported activation function {output_activation}.')

    return model


class KerasNeuralNetworks(object):

    def __init__(self, struct, n_epoch=100, lr=1e-4, batch_size=32, name='TEST', output_activation='softmax'):

        self._struct = [list(item) for item in struct]
        self._model = None
        self._learning_rate = lr
        self._batch_size = batch_size
        self._n_epoch = n_epoch
        self._name = name
        self._output_activation = output_activation

    def fit(self, X, y, show_accuracy=False):

        self._struct[0][0] = X.shape[1]

        X_ = np.array(X)
        y_ = np.array(y)
        if self._output_activation == 'softmax':
            y_ = y_.reshape(-1, 1)
            y_ = np.hstack([1 - y_, y_])

        self._model = KerasNeuralModel(struct=self._struct, lr=self._learning_rate,
                                       output_activation=self._output_activation)

        index_array = np.arange(X_.shape[0])
        batch_num = np.int(np.ceil(X_.shape[0] / self._batch_size))
        for i in np.arange(self._n_epoch):
            np.random.shuffle(index_array)
            for j in np.arange(batch_num):
                x_batch = X_[index_array[(j % batch_num) * self._batch_size:min((j % batch_num + 1) * self._batch_size,
                                                                                X_.shape[0])], :]
                if self._output_activation == 'softmax':
                    y_batch = y_[index_array[
                                 (j % batch_num) * self._batch_size:min((j % batch_num + 1) * self._batch_size,
                                                                        X_.shape[0])], :]
                elif self._output_activation == 'sigmoid':
                    y_batch = y_[index_array[
                                 (j % batch_num) * self._batch_size:min((j % batch_num + 1) * self._batch_size,
                                                                        X_.shape[0])]]
                else:
                    raise ValueError(f'Unsupported activation function {self._output_activation}.')
                self._model.train_on_batch(x_batch, y_batch)
            if (i + 1) % 150 == 0:
                K.set_value(self._model.optimizer.lr, K.eval(self._model.optimizer.lr * 0.1))
            acc = sum(self.predict(X).reshape(-1, ) == y) / len(y)

            print('training %s epoch %d / %d, accuracy=%.3f' % (self._name, i + 1, self._n_epoch, acc), end='\r')
            sys.stdout.flush()

    def predict(self, X):
        if self._output_activation == 'softmax':
            label_pred = self._model.predict(X)
            return (label_pred.reshape(-1)[:1] > 0).astype(int)
        elif self._output_activation == 'sigmoid':
            label_pred = self._model.predict(X)
            return (label_pred > 0).astype(int)
        else:
            raise ValueError(f'Unsupported activation function {self._output_activation}.')

    def predict_proba(self, X):
        if self._output_activation == 'softmax':
            label_pred = self._model.predict(X)
            return label_pred
        elif self._output_activation == 'sigmoid':
            label_pred = self._model.predict(X).reshape(-1, 1)
            return np.hstack([1 - label_pred, label_pred])
        else:
            raise ValueError(f'Unsupported activation function {self._output_activation}.')


class attack_model():
    def __init__(self,
                 num_epoch=60,
                 learning_rate=0.01,
                 batch_size=25,
                 verbose=1):
        self.a_model = tf.keras.Sequential([
            # tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            # tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.num_epoch = num_epoch
        self.verbos = verbose
        '''self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)'''
        self.lr_schedule = learning_rate
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.loss = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.a_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X_train, y_train):
        y_train = y_train.reshape(-1, 1)
        self.a_model.fit(X_train, y_train, epochs=self.num_epoch, verbose=self.verbos, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.a_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.a_model.predict(X)
        # y_pred = probs.argmax(axis=1)
        y_pred = probs.round()
        return y_pred


if __name__ == "__main__":
    # tf.executing_eagerly()
    FileNames = ['One_hot/ohAdult.csv',
                 'One_hot/ohCompas.csv',
                 'One_hot/ohBroward.csv',
                 'One_hot/ohHospital.csv'
                 ]
    lrs = [0.01, 0.005, 0.01, 0.01]
    # for i in range(4):
    for i in [0]:
        df = pd.read_csv(FileNames[i], header=None)
        data = np.array(df)
        input_size = data.shape[1] - 1
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.5)
        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        # test_dataset = tf .data.Dataset.from_tensor_slices((X_test, y_test)
        # model = target_model(num_epoch=30,num_microbatches=100, learning_rate=lrs[i], dp_flag=1, noise_multiplier=1.3)
        model = target_model(num_epoch=30, num_microbatches=50)
        model.fit(X_train, y_train)
        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)

        Train_Data = np.hstack((y_train.reshape(-1, 1),
                                train_probs,
                                np.ones([len(y_train), 1])))
        Test_Data = np.hstack((y_test.reshape(-1, 1),
                               test_probs,
                               np.zeros([len(y_test), 1])))

        AllData = np.vstack((Train_Data, Test_Data))
        mia_X_train, mia_X_test, mia_y_train, mia_y_test = train_test_split(AllData[:, :-1], AllData[:, -1],
                                                                            test_size=0.5)
        mia_label_train = mia_X_train[:, 0]
        mia_X_train = mia_X_train[:, 1:]
        mia_X_test = mia_X_test[:, 1:]
        model = KerasNeuralNetworks(
            struct=[(None, 512), (512, 256), (256, 128), (128, 1)],
            n_epoch=1000,
            batch_size=64,
            lr=1E-3,
            output_activation='sigmoid'
        )
        model.fit(mia_X_train, mia_y_train)

    print("pause")
