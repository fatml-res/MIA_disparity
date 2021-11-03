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

#from Calculate_ep import find_epoch
from NN_DP import target_model3
from Tools import find_all_results
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def run(args, t):
    datasets = ['Adult', 'Broward', 'Hospital']
    lr_list = [[5e-4, 1e-3, 1e-3], [0.001, 1e-3, 1e-3]]
    batch_ratio = [[80, 80, 100], [80, 80, 100]]
    epoch_ratio = [[3, 80, 5], [2, 80, 5]]

    for file_ind in args.file_list:  # range(len(FileNames)):
        lr = lr_list[1][file_ind]
        epoch_rate = epoch_ratio[1][file_ind]
        '''files = ["One_hot/ohAdult.csv",
                 "One_hot/ohBroward.csv",
                 "One_hot/ohHospital.csv"]'''
        files = ["One_hot/ohAdult.csv"]
        for file in files:
            if datasets[file_ind] in file:
                df = pd.read_csv(file, header=None)
                data = np.array(df)
                scaler = MinMaxScaler()
                data = scaler.fit_transform(data)
                X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.5)

                gender_train = X_train[:, 0]
                gender_test = X_test[:, 0]

                race_train = X_train[:, 1]
                race_test = X_test[:, 1]

                data_size = len(y_train)
                folder = "Original/t={}/".format(t) + datasets[file_ind]
                if os.path.exists(folder +'/target_result.csv'):
                    print("Skipped " + folder +'/target_result.csv')
                    continue
                print("Start Working on "+folder)

                batch_size = round(data_size / batch_ratio[1][file_ind]) - 1
                max_epoch = int(data_size * epoch_rate / 200)
                print("Max_epoch = ", max_epoch)
                delete_data = data_size % batch_size
                X_train_d = X_train[0:(data_size - delete_data), :]
                y_train_d = y_train[0:(data_size - delete_data)]
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
                save_result(model, X_train, X_test, y_train, y_test, gender_train, gender_test, race_train,
                            race_test, folder)


def save_result(model, X_train, X_test, y_train, y_test, gender_train, gender_test,
                race_train, race_test, folder, save_to_file=True):
    if not os.path.exists(folder) and save_to_file:
        os.makedirs(folder)
    prob_train = model.predict_proba(X_train)
    prob_test = model.predict_proba(X_test)
    Train_Data = np.hstack((y_train.reshape(-1, 1),
                            prob_train,
                            gender_train.reshape(-1, 1),
                            race_train.reshape(-1, 1),
                            np.ones([len(y_train), 1])))
    Test_Data = np.hstack((y_test.reshape(-1, 1),
                           prob_test,
                           gender_test.reshape(-1, 1),
                           race_test.reshape(-1, 1),
                           np.zeros([len(y_test), 1])))
    AllData = np.vstack((Train_Data, Test_Data))

    final_df = pd.DataFrame(AllData, index=None)
    if save_to_file:
        final_df.to_csv(folder + '/target_result.csv', header=False, index=False)

    return AllData


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--nm', type=float, nargs='+', default=[0.3], help='one or multiple noise multiplier')
    #parser.add_argument('--ep', type=float, nargs='+', default=[1.0], help='Epsilon')
    #parser.add_argument('-rep', type=int, nargs='+', default=4, help='number of repeating experimants')
    #parser.add_argument('--dp', type=int, nargs='+', default=[1], help='DP or not')
    parser.add_argument('--file_list', type=int, nargs='+', default=[0, 1, 2], help='which dataset')
    parser.add_argument('--verbos', type=int, default=2, help='verbos')
    parser.add_argument('--reduce', type=int, default=1, help='reduce learning rate or not')

    args = parser.parse_args()

    num_intra = tf.config.threading.get_intra_op_parallelism_threads()
    num_inter = tf.config.threading.get_inter_op_parallelism_threads()
    print("Current num_intra =", num_intra, "num_inter =", num_inter)

    tf.config.threading.set_intra_op_parallelism_threads(10)
    tf.config.threading.set_inter_op_parallelism_threads(10)
    for t in range(5):
        run(args, t)
