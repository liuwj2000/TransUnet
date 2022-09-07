import glob
import cv2
import numpy as np
#图像路径
path = r'.\raw_data\train\images\*.png'
#项目中存放训练所用的npz文件路径
path2 = r'.\data\Synapse\train_npz\\'
for i,img_path in enumerate(glob.glob(path)):
    	#读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
        label_path = img_path.replace('images','labels')
        label = cv2.imread(label_path,flags=0)
	#保存npz
        np.savez(path2+str(i),image=image,label=label)
        print('------------',i)

path = r'.\raw_data\val\images\*.png'
#项目中存放训练所用的npz文件路径
path2 = r'.\data\Synapse\test_vol_h5\\'
for i,img_path in enumerate(glob.glob(path)):
    	#读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
        label_path = img_path.replace('images','labels')
        label = cv2.imread(label_path,flags=0)
	#保存npz
        np.savez(path2+str(i),image=image,label=label)
        print('------------',i)

