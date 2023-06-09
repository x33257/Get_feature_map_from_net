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
import numpy as np
import sys
import time
import rsatoolbox as pyrsa


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
        for i in range(num_imgs-1):
            filepath_i = task_dat_name_list[n][i]
            dat_i = Load_result(filepath_i)
            assert len(dat_i) == num_layers, f'The Image {i} has a different number of model layers than the other images'
            keys = list(dat_i.keys())
            for j in range(i+1, num_imgs):
                filepath_j = task_dat_name_list[n][j]
                dat_j = Load_result(filepath_j)
                assert len(dat_j) == num_layers, f'The Image {i} has a different number of model layers than the other images'
                for d in range(num_layers):
                    RDM_for_layers[n, d, :, :] = np.eye(num_imgs)  # The correlation coefficient on the diagonal is 1.
                    RDM_for_layers[n, d, :, :] = np.corrcoef(dat_i[keys[d]].cpu().detach().numpy().ravel(),  # Only the upper triangular part is filled due to symmetry.
                                                   dat_j[keys[d]].cpu().detach().numpy().ravel())[0][1]
                    count = count+1
                    progress_bar(count, int((num_imgs-1)*num_imgs/2*num_layers), time.time()-start)
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
    return 1-6*(np.linalg.norm(x-y, ord=2)**2)/len(x)/(len(x)**2-1)

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
'''
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
        affinity_tensor[d, :, :] = np.eye(num_tasks)
        RDM_for_layers_triu_list = []
        for n in range(num_tasks):
            RDM_for_layers_triu_list.append(RDM_for_layers[n, d, :, :][np.triu_indices(num_imgs, 1)])  # Extract the upper triangle of each RDM to form a list.
        print('Upper triangular obtained!')
        for i in range(num_tasks-1):
            for j in range(i+1, num_tasks):
                affinity_tensor[d, i, j] = spearman_corr(RDM_for_layers_triu_list[i],
                                                          RDM_for_layers_triu_list[j])
                count = count+1
                progress_bar(count, int((num_tasks-1)*num_tasks/2*num_layers), time.time()-start)
    print('\n')
    print('Affinity tensor obtained!')
    return affinity_tensor
'''

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
    print(f'Getting {num_layers} RDMs of {num_tasks} tasks....')
    RDM_for_layers = []
    for n in range(num_tasks):
        print(f'Calculating on Task {n}...')
        count = 0
        start = time.time()
        tmp_list = []
        for d in range(num_layers):
            a_model = Load_result(task_dat_name_list[n][0])
            keys = list(a_model.keys())
            a_feature_map = a_model[keys[d]][0]
            input_mat = np.zeros((num_imgs, a_feature_map.shape[0]*a_feature_map.shape[1]*a_feature_map.shape[2]), dtype=np.float32)
            input_mat[0, :] = a_feature_map.cpu().detach().numpy().ravel()
            for k in range(num_imgs-1):
                a_model = Load_result(task_dat_name_list[n][k+1])
                a_feature_map = a_model[keys[d]][0]
                input_mat[k+1, :] = a_feature_map.cpu().detach().numpy().ravel()
            input_mat = pyrsa.data.Dataset(input_mat)
            RDM_for_a_layer = pyrsa.rdm.calc_rdm(input_mat, method='correlation')
            tmp_list.append(RDM_for_a_layer)
            count = count+1
            progress_bar(count, num_layers, time.time()-start)
        print('\n')
        RDM_for_layers.append(tmp_list)
        print(f'Task {n} finished!')
    print('RDMs obtained!')
    
    print('Getting an affinity tensor...')
    affinity_tensor = np.zeros((num_layers, num_tasks, num_tasks), np.float32)
    count = 0
    start = time.time()
    for d in range(num_layers):
        affinity_tensor[d, :, :] = np.eye(num_tasks)
        for i in range(num_tasks-1):
            for j in range(i+1, num_tasks):
                affinity_tensor[d, i, j] = pyrsa.rdm.compare_spearman(RDM_for_layers[i][d], RDM_for_layers[j][d])[0][0]
                count = count+1
                progress_bar(count, int((num_tasks-1)*num_tasks/2*num_layers), time.time()-start)
    print('\n')
    print('Affinity tensor obtained!')
    return affinity_tensor


if __name__=='__main__':
    task_dat_path_list = ['C:\exp\loc', 'C:\exp\conf']
    
    affinity_tensor = get_affinity_tensor(task_dat_path_list)
    with open('res.txt', 'w') as f:
        for d in range(affinity_tensor.shape[0]):
            print(affinity_tensor[d][0][1])
            f.write('\n', affinity_tensor[d][0][1])
    f.close()