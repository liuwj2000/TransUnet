import os

f1=open(r'.\lists\lists_Synapse\test_vol.txt','w')
for i in os.listdir(r'.\data\Synapse\test_vol_h5'):
    f1.write(os.path.splitext(i)[0])
    f1.write('\n')
#f1.write('\b')
f1.close()
    
f2=open(r'.\lists\lists_Synapse\train.txt','w')
for i in os.listdir(r'.\data\Synapse\train_npz'):
    f2.write(os.path.splitext(i)[0])
    f2.write('\n')
#f1.write('\b')
f2.close()
