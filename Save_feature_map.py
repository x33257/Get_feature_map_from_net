import pickle
import os.path


def Save_Fmap2dic(dic,data,layer_name):
    data_dic=dic
    data_dic[layer_name]=data
    return dic

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def Save_result(path_save,name_save,data):
    if not os.path.exists(path_save):
        mkdirs(path_save)
    path=os.path.join(path_save,name_save)
    with open(path,'wb') as f:
        pickle.dump(data,f)

def Load_result(path_load):
    with open(path_load,'rb') as f:
        data=pickle.load(f)
   #print(data)
    return data

def Get_name_from_path(path):
    x=path.split('/')
    y=x[-1]
    name=y.split('.')
    return name[0]

if __name__=='__main__':
    path_load='./FeatureMap_MSRS_VGG03_ssd_vi/00095D_vi_VGGnet_after_CNN3.dat'
    data=Load_result(path_load)
    for i in data:
        print(i)
        print(data[i].shape)
    # CnnFusNet 融合后的结果是304*304。 只取0：299 即可

    # path='/home/host/workspace_sda/workpace_all/python_workspace/Net_Base/test_CNN.py'
    # print(Get_name_from_path(path))