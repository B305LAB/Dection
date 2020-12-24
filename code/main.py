# coding: utf-8
import os
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from keras.models import Sequential,Model
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose,Concatenate,Input
import matplotlib.pyplot as plt
import warnings
import json
import cv2
import zipfile
warnings.filterwarnings("ignore")
train_path1 = "../raw_data/round_train/part1/OK_Images/"

test_path1 =  "../raw_data/round_test/part1/TC_Images/" #产品1的测试集路径

model_path1 =  '../model/model1/'

temp_path = '../temp_data/'

save_path1 = '../temp_data/result/data/focusight1_round2_train_part1/TC_Images/'

result_path = '../result/'

mask_save = '../temp_data/'

'''构建模型'''
model = Sequential([
    Conv2D(filters=30, kernel_size=[3, 3], strides=[2, 2], padding='same', input_shape=[128, 128, 1],
           activation='relu'),
    Conv2D(filters=60, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 64*64*30
    Conv2D(filters=90, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 32*32*60
    Conv2D(filters=120, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 16*16*90
    Conv2DTranspose(90, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # 32x32x120
    Conv2DTranspose(60, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  ## # 32x32x60
    Conv2DTranspose(30, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # # 32x32x30
    Conv2DTranspose(1, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu'),  # # 32x32x1
])
model.summary()
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
''''''''''''''''''''''''''''''''


def load_data(test_path):
    testfiles = os.listdir(test_path)
    testfile_names = []
    test = np.empty([len(testfiles),128,128,1])
    for i in range(len(testfiles)):
        testfile_names.append(os.path.splitext(testfiles[i])[0])
        img = cv2.imread(test_path + testfiles[i],cv2.IMREAD_GRAYSCALE).reshape([128,128,1])
        test[i] = img
    return test, testfile_names


def Dection(X, Y,file_name):
    if X.ndim==4:
        X = X[:,:,:,0]
        Y = Y[:,:,:,0]
    img_diff = np.abs(X-Y)
    res = np.copy(img_diff)
    for i in range(0,len(X)):
        img1 = img_diff[i] # 阈值大概在10左右
        img3 = np.copy(img1)
        img3[img3<18] = 0
        img3[img3>=18] = 255
        res[i] = img3
        x_origin = np.zeros(shape=[*X[i].shape,3],dtype=np.uint8)
        x_origin[:,:,0]=X[i]
        x_origin[:,:,1]=X[i]
        x_origin[:,:,2]=X[i]
        x_mask = np.copy(x_origin)
        x_mask[img3>0,:]=(255,0,0)
        plt.imsave(temp_path+'mask/{}.jpg'.format(file_name[i]),np.concatenate([x_mask,x_origin],axis=0),vmin = 0,vmax = 255)
    return res
def get_segment_result(test, testfile_names, model_path):
    '''首先加载模型'''
    model.load_weights(model_path + 'model.h5')
    test = test.astype('float32')
    '''再对模型进行预测'''
    y_test = model.predict(test)
    '''对预测的结果进行处理'''
    mask_imgs = Dection(test, y_test,testfile_names)
    regions = []
    for i in range(len(mask_imgs)):
        img = mask_imgs[i]
        index = np.argwhere(img>0)
        region = []
        for j in range(len(index)):
            x, y = index[j]
            region.append("{}, {}".format(x,y))
        regions.append(region)
    region_df = pd.DataFrame({'regions': regions,
                            'testfile_names': testfile_names}) 
    return region_df


#本示例未进行像素级的检测，为了保证result.zip的完整性，示例中人为写入一对像素坐标
def save_result(tc_images,save_path):
    for row_index,row in tc_images.iterrows():
        result = {}
        result['Height'] = 128
        result['Width'] = 128
        result['name'] = row['testfile_names'] + '.bmp'
        regions = []
        points = {}
        points['points'] = row['regions'] #请注意，两像素值中间有一个空格
        regions.append(points)
        result['regions'] = regions
        if len(row['regions']) < 10 or len(row['regions']) > 3000:
            continue
        savefile = save_path + row['testfile_names'] + '.json'
        with open(savefile,'w') as f:
            f.write(json.dumps(result,ensure_ascii=False,indent=2))   

def zipdir(path,file):
    z = zipfile.ZipFile(file,'w',zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(path):
        fpath = dirpath.replace(path,'') 
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()


test1, testfile_names1 = load_data(test_path1)

region_df1 = get_segment_result(test1, testfile_names1, model_path1)

save_result(region_df1,save_path1)

zipdir(temp_path + 'result/', result_path + 'data.zip')

