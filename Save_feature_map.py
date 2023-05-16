import pickle
import cv2
import os
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


def Save_FeatureMap_image(feature_map,layername:str):
    feature_size=feature_map.shape  #B,C,H,W  in most decase B==1
    img = tensor_to_mat(feature_map)
    for batch in range(feature_size[0]):
        save_dir = rf'./FeatureMap_Image/batch{batch}/{layername}'
        os.makedirs(save_dir, exist_ok=True)
        for n in range(feature_size[1]):
            image_name = rf'{save_dir}/{n}.jpg'
            cv2.imwrite(image_name,img[:,:,n])

def tensor_to_mat(img_tensor):
    img_tensor1=img_tensor.cpu().permute(0,2,3,1).contiguous().detach().numpy()  #to N*C*H*W
    size_np=img_tensor1.shape
    img_mat=(img_tensor1[0:,:,:]*255).reshape(size_np[1],size_np[2],size_np[3]).astype('uint8')
    return img_mat


def Save_image_from_dat(dat_path):
    data_in_dat=Load_result(dat_path)
    print('Get the data now! They are:')
    for i in data_in_dat:
        print(i)
        print(data_in_dat[i].shape)
        Save_FeatureMap_image(data_in_dat[i],i)




if __name__=='__main__':
    path_load='/home/host/data/test02/Feature_map_dat'+'/FeatureMap_MSRS_VGG03_ssd_ir/00095D_ir_VGGnet_after_CNN3.dat'
    Save_image_from_dat(path_load)


