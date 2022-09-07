import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path='./prediction', case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    #print('im,la',image.shape,label.shape) (3.753.1600) (753,1600)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        print(prediction.shape)
        #print(image.shape[0]) #3
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1] 
            #print('x,y',x,' ',y) #753 1600
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            #print('slice',slice.shape) (224,224)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda('cuda:2')
            #print('input',input.shape) (1,1,224,224)
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                #print('output',outputs.shape) (1,9,224,224)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                #print('out',out.shape) (224,224)
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                #print(pred[pred!=0])
                #prediction[ind] = pred
                prediction+= pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda('cuda:2')
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    #print(prediction)
    np.save(case+'.npy',prediction)
    plt.imshow(prediction)
    plt.savefig(case+'.png')
    #print(image.shape,prediction.shape,label.shape)
    
    image1=image.reshape(image.shape[1],image.shape[2],image.shape[0])
    
    image2=image1
    #print(image2.shape)
    #image2= cv2.cvtColor(image2,cv2.COLOR_GRAY2BGR)
    #print(image2.shape)
    #print('where',image2[np.where(prediction==0)].shape)
    image2 = np.where(prediction==0, np.full_like(image2, blue ), image2)
    image2 = np.where(prediction==1, np.full_like(image2, green  ), image2)
    image2 = np.where(prediction==2, np.full_like(image2, red  ), image2)

    print(image.shape,prediction.shape,label.shape)
    print('trs',test_save_path)
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        print(test_save_path)
        sitk.WriteImage(prd_itk, test_save_path + '/'+ "pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ "img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + "gt.nii.gz")
    return metric_list