import sys

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def bi_search(min_nm, max_nm, target_ep, n=10000, batch_size=1, epoch=60):
    mid_nm = (min_nm + max_nm) / 2
    mid_nm_ep = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=n,
                                                              batch_size=batch_size,
                                                              noise_multiplier=mid_nm,
                                                              epochs=epoch,
                                                              delta=1e-5)[0]
    finish_flag = True
    while finish_flag:
        if mid_nm_ep < target_ep - 0.001:  # ep too small, nm too large
            max_nm = mid_nm
        elif mid_nm_ep > target_ep + 0.001:  # ep too large, nm too small
            min_nm = mid_nm
        else:
            finish_flag = False
        mid_nm = (max_nm + min_nm) / 2
        mid_nm_ep = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=n,
                                                                  batch_size=batch_size,
                                                                  noise_multiplier=mid_nm,
                                                                  epochs=epoch,
                                                                  delta=1e-5)[0]
    return mid_nm


def find_epoch(s_max_epoch, nm, n, batch_size, ep_list=[3.0, 1.0, 0.5, 0.3, 0.1]):
    epoch_list = [s_max_epoch]
    for ep in ep_list:
        min_epoch = 1
        max_epoch = int(s_max_epoch)
        mid_epoch = int((min_epoch + max_epoch) / 2)

        ep_mid = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=n,
                                                               batch_size=batch_size,
                                                               noise_multiplier=nm,
                                                               epochs=mid_epoch,
                                                               delta=1e-5)[0]
        searching_flag = True
        sys.stdout.flush()
        while searching_flag:
            if ep_mid > ep + 0.1:  # mid ep is too large, need smaller epoch
                max_epoch = mid_epoch
            elif ep_mid < ep - 0.1:
                min_epoch = mid_epoch
            else:
                searching_flag = False
            mid_epoch = int((max_epoch + min_epoch) / 2)
            ep_mid = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=n,
                                                                   batch_size=batch_size,
                                                                   noise_multiplier=nm,
                                                                   epochs=mid_epoch,
                                                                   delta=1e-5)[0]
            sys.stdout.flush()
        epoch_list.append(mid_epoch)
    epoch_list.sort()
    ep_list = [5.0] + ep_list
    ep_list.sort()
    return ep_list, epoch_list


if __name__ == "__main__":
    tmp = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=3600,
                                                        batch_size=100,
                                                        noise_multiplier=20,
                                                        epochs=14400,
                                                        delta=1e-5)[0]
    print("Pause")
    # find_epoch(2705, 8.65, n=3607, batch_size=90, ep_list=[3.0, 2.0, 1.0,  0.7, 0.5, 0.3])
    '''target_ep = [0.01,0.05,0.1,0.5,1,5]
    nm_list = []
    for ep in target_ep:
        nm = bi_search(0.1, 100000, ep)
        nm_list.append(nm)
    print(target_ep)
    print(nm_list)'''

'''for delta in delta_list:
    ep=compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=20000,
                                                     batch_size=25,
                                                     noise_multiplier=1.3,
                                                     epochs=30,
                                                     delta=delta)[0]
    ep_list.append(ep)
for i in range(len(ep_list)):
    print('delta='+str(delta_list[i])+', epsilon=' +str(ep_list[i]))'''
