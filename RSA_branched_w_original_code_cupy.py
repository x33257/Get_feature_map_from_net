"""An implementation of the RSA algorithm from Branched Multi-Task Networks:
Deciding What Layers To Share.

Given the folder task_dat_path_list where .dat files of each task are located,
solve for the affinity tensor.

Examples:

task_dat_path_list = ['C:\exp\loc', 'C:\exp\conf']
affinity_tensor = get_affinity_tensor(task_dat_path_list)
"""

from Save_feature_map import Load_result
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cupy as np
import numpy
import sys
import time


def progress_bar(finished_count, total_count, accumulated_time):
    scale = finished_count/total_count
    percentage = int(scale*100)
    finished_str = "▋"*(percentage//2)
    unfinished_str = '.'*(50-percentage//2)
    ETA = accumulated_time*(1-scale)/scale
    print(f"Progress: {percentage}% ({finished_count}/{total_count})", f'[{finished_str}->{unfinished_str}]{accumulated_time:.2f}/ETA: {ETA:.2f}', end="\r")
    sys.stdout.flush()

def get_dat_name(path_list):
    """Get a list of .dat file names for each task from the path_list.

    Args:
        path_list: list
          A folder where .dat files of each task are located.

    Returns:
        A list where each element contains a list of .dat file names for each task.
    """
    print('Getting lists of filenames...')
    task_dat_name_list = []
    for path in path_list:
        temp_list = []
        for filename in os.listdir(path):
            temp_list.append(os.path.join(path, filename))
        temp_list.sort()
        task_dat_name_list.append(temp_list)
    print('Finished!')
    return task_dat_name_list

def get_RDM(num_layers, num_imgs, num_tasks, task_dat_name_list):
    """Get a ndarray of RDM of num_tasks tasks using given parameters.

    Args:
        num_layers: int
          The number of Locations in the network.
        num_imgs: int
          The number of input images.
        num_tasks: int
          The number of tasks.
        task_dat_name_list: list
          A list where each element contains a list of .dat file names for each task.

    Returns:
        A ndarray of RDM of N tasks with dimensions (num_tasks, num_layers, num_imgs, num_imgs).
    """
    print(f'Getting {num_layers} RDMs of {num_tasks} tasks....')
    RDM_for_layers = np.zeros((num_tasks, num_layers, num_imgs, num_imgs), np.float32)
    for n in range(num_tasks):
        print(f'Calculating on Task {n}...')
        count = 0
        start = time.time()
        for d in range(num_layers):
            all_features_for_img0 = Load_result(task_dat_name_list[n][0])
            assert len(all_features_for_img0) == num_layers, f'The Image 0 has a different number of model layers than the other images'
            keys = list(all_features_for_img0.keys())
            feature_0 = all_features_for_img0[keys[d]].cpu().detach().numpy().ravel()
            features_value = np.zeros((num_imgs,feature_0.shape[0]))
            for i in range(num_imgs):
                feature = np.asarray(Load_result(task_dat_name_list[n][i])[keys[d]].cpu().detach().numpy())
                features_value[i,:] = feature.ravel()
            features_value = np.asnumpy(features_value)
            features_value = features_value - numpy.mean(features_value, axis=0)
            RDM_for_layers[n, d, :, :] = np.asarray(1-numpy.corrcoef(features_value))
            count = count+1
            progress_bar(count, num_layers, time.time()-start)
        print('\n')
        print(f'Task {n} finished!')
    print('RDMs obtained!')
    return RDM_for_layers

def spearman_corr(x, y):
    """Calculate Spearman correlation coefficient of two vectors x and y.

    Args:
        x: array_like
          A vector.
        y: array_like
          A vector.

    Returns:
        A Spearman correlation coefficient.
    """
    return 1-6*(np.linalg.norm(x-y, ord=2)**2)/len(x)/(len(x)**2-1)\

def fast_spearman_corr(array):
    num_tasks = array.shape[0]
    res = np.zeros((num_tasks, num_tasks))
    tmp1 = np.tile(array[0],(num_tasks-1,1))
    tmp2 = array[1:]
    for i in range(num_tasks-2):
        tmp1 = np.append(tmp1, np.tile(array[i+1], (num_tasks-2-i, 1)), axis=0)
        tmp2 = np.append(tmp2, array[i+2:], axis=0)
    res[np.triu_indices(num_tasks, 1)] = 1-6*(np.linalg.norm(tmp1-tmp2, ord=2, axis=1)**2)/array.shape[1]/(array.shape[1]**2-1)
    return res

def feature_maps_pearson(dat_path1, dat_path2):
    dat_1 = Load_result(dat_path1)
    dat_2 = Load_result(dat_path2)
    num_layers = len(dat_1)
    assert num_layers == len(dat_2), 'Different number of model layers.'
    res_vec = np.zeros((num_layers), np.float32)
    keys = list(dat_1.keys())
    start = time.time()
    for d in range(num_layers):
        res_vec[d] = np.corrcoef(dat_1[keys[d]].cpu().detach().numpy().ravel(), dat_2[keys[d]].cpu().detach().numpy().ravel())[0][1]
        progress_bar(d+1, num_layers, time.time()-start)
    return res_vec

def get_affinity_tensor(task_dat_path_list):
    """Get a ndarray of affinity tensor using given parameters.

    Args:
        num_layers: int
          The number of Locations in the network.
        num_imgs: int
          The number of input images.
        num_tasks: int
          The number of tasks.
        task_dat_path_list: list
          A list where .dat files of each task are located.

    Returns:
        A ndarray of affinity tensor with dimensions (num_layers, num_tasks, num_tasks).
    """
    assert len(task_dat_path_list) >= 2, 'Need to compare at least two tasks.'
    num_tasks = len(task_dat_path_list)
    print(f'Need to calculate the similarity of {num_tasks} tasks.')
    task_dat_name_list = get_dat_name(task_dat_path_list)
    num_imgs = len(task_dat_name_list[0])
    for n in range(num_tasks):
        assert num_imgs == len(task_dat_name_list[n]), f'The number of images for Task {n} is different from that for other tasks.'
    print(f'Each task has {num_imgs} input images.')
    num_layers = len(Load_result(task_dat_name_list[0][0]))
    assert num_layers != 0, 'Empty layers.'
    print(f'Each model has {num_layers} layers of feature maps output locations.')
    RDM_for_layers = get_RDM(num_layers, num_imgs, num_tasks, task_dat_name_list)
    print('Getting an affinity tensor...')
    affinity_tensor = np.zeros((num_layers, num_tasks, num_tasks), np.float32)
    count = 0
    start = time.time()
    for d in range(num_layers):
        RDM_for_layers_triu = np.zeros((num_tasks, num_imgs*(num_imgs-1)//2), dtype=np.float32)
        for n in range(num_tasks):
            RDM_for_layers_triu[n, :] = RDM_for_layers[n, d, :, :][np.triu_indices(num_imgs, 1)]  # Extract the upper triangle of each RDM to form a list.
        affinity_tensor[d, :, :] = fast_spearman_corr(RDM_for_layers_triu)
        count = count+1
        progress_bar(count, num_layers, time.time()-start)
    print('\n')
    print('Affinity tensor obtained!')
    return affinity_tensor

if __name__=='__main__':
    task_dat_path_list = ['C:\exp\loc', 'C:\exp\conf']
    affinity_tensor = get_affinity_tensor(task_dat_path_list)
    for d in range(affinity_tensor.shape[0]):
        print(affinity_tensor[d][0][1])