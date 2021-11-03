'''
Use one hot datasets to train N target models, save raw_data+prob+train_test_label
to target_result/dataset_name/nm=?/time=N/
'''

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from Calculate_ep import find_epoch

#from Calculate_ep import find_epoch
from NN_DP import target_model3
from Tools import find_all_results
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def divide_train_test_by_group(data):
    r = (data[:, 1]==0).sum()/(data[:, 1] == 1).sum()
    all_groups = np.unique(data[:, 0])
    up_ind = data[:, 1] == 0
    p_ind = data[:, 1] == 1
    groups = data[:, 0]

    g_train, g_test, y_train, y_test = train_test_split(data[:, 0], data[:, -1], test_size=0.5)
    data_train = []
    data_test = []
    for g in all_groups:
        g_ind = groups == g
        g_train_count = (g_train == g).sum()
        count_up_g_train = round(r*g_train_count/(1+r))
        data_up_g = data[g_ind*up_ind]
        np.random.shuffle(data_up_g)
        tmp_up_train = data_up_g[:count_up_g_train]
        data_train.append(tmp_up_train)
        tmp_up_test = data_up_g[count_up_g_train:]
        data_test.append(tmp_up_test)

        count_p_g_train = g_train_count - count_up_g_train
        data_p_g = data[g_ind*p_ind]
        np.random.shuffle(data_p_g)
        tmp_p_train = data_p_g[:count_p_g_train]
        tmp_p_test = data_p_g[count_p_g_train:]
        data_train.append(tmp_p_train)
        data_test.append(tmp_p_test)
        # print("In group {}, there are {} up data and {} p data in training,
        # {} up data and {} p data for testing".format(g, len(tmp_up_train),
        # len(tmp_p_train), len(tmp_up_test), len(tmp_p_test)))
    data_train = np.vstack(data_train)
    np.random.shuffle(data_train)

    data_test = np.vstack(data_test)
    np.random.shuffle(data_test)
    return data_train[:, :-1], data_test[:, :-1], data_train[:, -1], data_test[:, -1]


def load_data(data):
    tr_te_label = data[:, -1]
    train_ind = tr_te_label == 1
    test_ind = tr_te_label == 0
    g_train = data[train_ind, 0]
    g_test = data[test_ind, 0]
    X_train = data[train_ind, 1:-2]
    X_test = data[test_ind, 1:-2]
    y_train = data[train_ind, -2]
    y_test = data[test_ind, -2]
    return g_train.reshape(-1, 1), g_test.reshape(-1, 1), X_train, X_test, y_train, y_test

def run(args):
    datasets = ['Adult', 'Broward', 'Hospital']
    lr_list = [[1e-3, 1e-3, 1e-3], [[0.001, 0.001], [1e-3, 0.001], [1e-3, 1e-3]]]
    batch_ratio = [[80, 80, 100], [[80, 80], [80, 80], [100, 100]]]
    epoch_ratio = [[4, 80, 5], [[4, 10], [80, 80], [5, 5]]]

    for file_ind in args.file_list:  # range(len(FileNames)):
        #files = find_all_results("Fairpick/FairPick/"+datasets[file_ind], result_type="Fairpick_data.csv")
        lr = lr_list[0][file_ind]
        epoch_rate = epoch_ratio[0][file_ind]
        files = find_all_results("Fairpick-test/", "Oh_data.csv")
        for file in files:
            if datasets[file_ind] in file:
                df = pd.read_csv(file, header=None)
                data = np.array(df)
                if -1 in data[:, 0]:
                    print("Found a Wrong dataset in {}".format(file))
                    continue
                if os.path.exists(file.replace("Oh_data", "target_result")):
                    print("Skipped " + file.replace("Oh_data", "target_result"))
                    continue
                satt = file.split('/')[-2]
                if float(file.split('/')[1].replace("n=", ""))<0:
                    print("Skipped file {} because it is running on another server!".format(file))
                    continue


                data = data[data[:, 0]>-1]
                scaler = MinMaxScaler()
                data[:, 1:] = scaler.fit_transform(data[:, 1:])
                # X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.5)
                X_train, X_test, y_train, y_test = divide_train_test_by_group(data)
                if args.with_group:
                    group_train = X_train[:, 0].reshape(-1, 1)
                    X_train = X_train[:, 1:]
                    group_test = X_test[:, 0].reshape(-1, 1)
                    X_test = X_test[:, 1:]
                    if len(np.unique(group_train))==1 or len(np.unique(group_test))==1:
                        print("Only one cluster exist in file{}".format(file))
                        continue
                else:
                    group_train = []
                    group_test = []

                s_train = X_train[:, 0]
                s_test = X_test[:, 0]
                data_size = len(y_train)
                batch_size = round(data_size / batch_ratio[0][file_ind]) - 1
                max_epoch = int(data_size * epoch_rate / 200)
                print("Max_epoch = ", max_epoch)
                delete_data = data_size % batch_size
                X_train_d = X_train[0:(data_size - delete_data), :]
                y_train_d = y_train[0:(data_size - delete_data)]
                if args.dp[0] == 0:
                    print("Start working on file {}".format(file))
                    #print("Start Working on %s dataset on sensitive attr is %s" % (datasets[file_ind], satt))

                    model = target_model3(num_epoch=max_epoch,
                                          dp_flag=0,
                                          l2_norm_clip=30,
                                          noise_multiplier=1,
                                          num_microbatches=batch_size,
                                          learning_rate=lr,
                                          data_size=data_size,
                                          verbos=args.verbos,
                                          reduce=args.reduce)
                    model.fit(X_train_d, y_train_d, epoch=max_epoch)
                    folder = file.split("Oh_data.csv")[0]
                    save_result(model, X_train, X_test, y_train, y_test,
                                s_train, s_test, group_train, group_test, folder)
                else:
                    if satt == 'gender':
                        sa_ind = 0
                    else:
                        sa_ind = 1

                    lr = lr_list[1][file_ind][sa_ind]
                    epoch_rate = epoch_ratio[1][file_ind][sa_ind]
                    batch_size = round(data_size / batch_ratio[1][file_ind][sa_ind]) - 1
                    max_epoch = int(data_size * epoch_rate / 200)
                    print("Max_epoch = ", max_epoch)
                    delete_data = data_size % batch_size
                    X_train_d = X_train[0:(data_size - delete_data), :]
                    y_train_d = y_train[0:(data_size - delete_data)]


                    if os.path.exists(file.replace("Oh_data", "DP/ep=5.0/target_result")):
                        print("Skipped " + file.replace("Oh_data", "DP/ep=5.0/target_result"))
                        continue
                    print("Start Working on {} dataset on sensitive attr is {} with DP".format(datasets[file_ind], satt))
                    ep = 5.0
                    nm = compute_noise(data_size, batch_size, ep,
                                       max_epoch, 1e-5, 1e-6)
                    eps, epoch_list = find_epoch(max_epoch, nm, data_size, batch_size, ep_list=[0.3, 0.5, 1.0, 3.0])
                    model = target_model3(num_epoch=max_epoch,
                                          dp_flag=1,
                                          l2_norm_clip=30,
                                          noise_multiplier=nm,
                                          num_microbatches=batch_size,
                                          learning_rate=lr,
                                          data_size=data_size,
                                          verbos=args.verbos,
                                          reduce=args.reduce)
                    tmp_epoch = 0
                    for ep_ind in range(len(epoch_list)):
                        print("Epsilon=", eps[ep_ind])
                        model.fit(X_train_d, y_train_d, epoch=epoch_list[ep_ind] - tmp_epoch)
                        tmp_epoch = epoch_list[ep_ind]
                        folder = file.replace("Oh_data.csv", "DP/ep={}/".format(eps[ep_ind]))
                        save_result(model, X_train, X_test, y_train, y_test, s_train, s_test, [], [], folder)


def run_v2(args):
    datasets = ['Adult', 'Broward', 'Hospital']
    lr_list = [[1e-3, 1e-3, 1e-3], [[0.001, 0.001], [1e-3, 0.001], [1e-3, 1e-3]]]
    batch_ratio = [[80, 80, 100], [[80, 80], [80, 80], [100, 100]]]
    epoch_ratio = [[4, 80, 5], [[4, 10], [80, 80], [5, 5]]]

    for file_ind in args.file_list:
        lr = lr_list[0][file_ind]
        epoch_rate = epoch_ratio[0][file_ind]
        files = find_all_results("Fairpick-bound/", "Oh_data_1.csv")
        for file in files:
            if datasets[file_ind] in file:
                df = pd.read_csv(file, header=None)
                data = np.array(df)
                if os.path.exists(file.replace("Oh_data_1", "target_result_1")):
                    print("Skipped " + file.replace("Oh_data_1", "target_result_1"))
                    continue
                satt = file.split('/')[-2]
                if satt == "Gender":
                    print("Skip Gender for {}".format(file))
                    continue
                if float(file.split('/')[1].replace("n=", ""))<0.12:
                    print("Skipped file {} because it is running on another server!".format(file))
                    continue


                data = data[data[:, 0]>-1]
                scaler = MinMaxScaler()
                data[:, 1:] = scaler.fit_transform(data[:, 1:])
                group_train, group_test, X_train, X_test, y_train, y_test = load_data(data)

                s_train = X_train[:, 0]
                s_test = X_test[:, 0]
                data_size = len(y_train)
                batch_size = round(data_size / batch_ratio[0][file_ind]) - 1
                max_epoch = int(data_size * epoch_rate / 200)
                print("Max_epoch = ", max_epoch)
                delete_data = data_size % batch_size
                X_train_d = X_train[0:(data_size - delete_data), :]
                y_train_d = y_train[0:(data_size - delete_data)]
                if args.dp[0] == 0:
                    print("Start working on file {}".format(file))
                    #print("Start Working on %s dataset on sensitive attr is %s" % (datasets[file_ind], satt))

                    model = target_model3(num_epoch=max_epoch,
                                          dp_flag=0,
                                          l2_norm_clip=30,
                                          noise_multiplier=1,
                                          num_microbatches=batch_size,
                                          learning_rate=lr,
                                          data_size=data_size,
                                          verbos=args.verbos,
                                          reduce=args.reduce)
                    model.fit(X_train_d, y_train_d, epoch=max_epoch)
                    folder = file.split("Oh_data_1.csv")[0]
                    save_result(model, X_train, X_test, y_train, y_test,
                                s_train, s_test, group_train, group_test, folder)
                else:
                    if satt == 'gender':
                        sa_ind = 0
                    else:
                        sa_ind = 1

                    lr = lr_list[1][file_ind][sa_ind]
                    epoch_rate = epoch_ratio[1][file_ind][sa_ind]
                    batch_size = round(data_size / batch_ratio[1][file_ind][sa_ind]) - 1
                    max_epoch = int(data_size * epoch_rate / 200)
                    print("Max_epoch = ", max_epoch)
                    delete_data = data_size % batch_size
                    X_train_d = X_train[0:(data_size - delete_data), :]
                    y_train_d = y_train[0:(data_size - delete_data)]


                    if os.path.exists(file.replace("Oh_data_1", "DP/ep=5.0/target_result_1")):
                        print("Skipped " + file.replace("Oh_data", "DP/ep=5.0/target_result"))
                        continue
                    print("Start Working on {} dataset on sensitive attr is {} with DP".format(datasets[file_ind], satt))
                    ep = 5.0
                    nm = compute_noise(data_size, batch_size, ep,
                                       max_epoch, 1e-5, 1e-6)
                    eps, epoch_list = find_epoch(max_epoch, nm, data_size, batch_size, ep_list=[0.3, 0.5, 1.0, 3.0])
                    model = target_model3(num_epoch=max_epoch,
                                          dp_flag=1,
                                          l2_norm_clip=30,
                                          noise_multiplier=nm,
                                          num_microbatches=batch_size,
                                          learning_rate=lr,
                                          data_size=data_size,
                                          verbos=args.verbos,
                                          reduce=args.reduce)
                    tmp_epoch = 0
                    for ep_ind in range(len(epoch_list)):
                        print("Epsilon=", eps[ep_ind])
                        model.fit(X_train_d, y_train_d, epoch=epoch_list[ep_ind] - tmp_epoch)
                        tmp_epoch = epoch_list[ep_ind]
                        folder = file.replace("Oh_data.csv", "DP/ep={}/".format(eps[ep_ind]))
                        save_result(model, X_train, X_test, y_train, y_test, s_train, s_test, [], [], folder)



def save_result(model, X_train, X_test, y_train, y_test, s_train, s_test, g_train, g_test, folder):
    if os.path.exists(folder + "/target_result_1.csv"):
        print("Skipped " + folder)
        return 0
    if not os.path.exists(folder):
        os.makedirs(folder)
    prob_train = model.predict_proba(X_train)
    prob_test = model.predict_proba(X_test)
    Train_Data = np.hstack((g_train,
                            y_train.reshape(-1, 1),
                            prob_train,
                            s_train.reshape(-1, 1),
                            np.ones([len(y_train), 1])))
    Test_Data = np.hstack((g_test,
                           y_test.reshape(-1, 1),
                           prob_test,
                           s_test.reshape(-1, 1),
                           np.zeros([len(y_test), 1])))
    AllData = np.vstack((Train_Data, Test_Data))

    final_df = pd.DataFrame(AllData, index=None)
    final_df.to_csv(folder + '/target_result_1.csv', header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--nm', type=float, nargs='+', default=[0.3], help='one or multiple noise multiplier')
    #parser.add_argument('--ep', type=float, nargs='+', default=[1.0], help='Epsilon')
    #parser.add_argument('-rep', type=int, nargs='+', default=4, help='number of repeating experimants')
    parser.add_argument('--dp', type=int, nargs='+', default=[0], help='DP or not')
    parser.add_argument('--file_list', type=int, nargs='+', default=[0, 1, 2], help='which dataset')
    parser.add_argument('--verbos', type=int, default=2, help='verbos')
    parser.add_argument('--reduce', type=int, default=1, help='reduce learning rate or not')
    parser.add_argument('--with_group', type=bool, default=True, help="The first column is group or not")

    args = parser.parse_args()

    num_intra = tf.config.threading.get_intra_op_parallelism_threads()
    num_inter = tf.config.threading.get_inter_op_parallelism_threads()
    print("Current num_intra =", num_intra, "num_inter =", num_inter)

    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
    run_v2(args)
