#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):

    '''
    准确度，像素值直接比较
    和真实值相同为1，不同为0
    输出的是 正确标记的像素/总体像素
    '''
    
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    corr = torch.sum((SR == GT).int())
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    # print("SR.size:%d, %d, %d, %d " %(SR.size(0),SR.size(1),SR.size(2),SR.size(3)))
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    '''
    正确标记的道路像素 / 真实的道路像素
    '''
    SR = SR > threshold
    # print('GT: ',GT)
    # print('torch.max(GT): ',torch.max(GT))
    GT = GT == torch.max(GT)
    SR = SR.int()
    GT = GT.int()

    # TP : True Positive
    # FN : False Negative

    TP = ((SR + GT) == 2).int()
    FN = (((SR == 0).int()+(GT == 1).int()) == 2).int()
    # print('torch.sum(TP): ',torch.sum(TP))
    # print('torch.sum(GT): ',torch.sum(GT))
    SE = float(torch.sum(TP))/(float(torch.sum(GT)) + 1e-6)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    '''
    正确标记的非道路像素 / 真实非道路像素
    '''
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    # TN : True Negative
    # FP : False Positive
    TN = ((SR + GT) == 0).int()
    FP = (((SR==1).int()+(GT==0).int())==2).int()

    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    SP = float(torch.sum(TN))/(float(tensor_size - torch.sum(GT)) + 1e-6)
    

    return SP


def get_precision(SR,GT,threshold=0.5):
    '''
    正确标记的道路像素 / 网络输出的道路像素
    '''
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    # TP : True Positive
    # FP : False Positive
    TP = ((SR + GT) == 2).int()
    FP = (((SR==1).int()+(GT==0).int())==2).int()


    PC = float(torch.sum(TP))/(float( torch.sum(SR)) + 1e-6)

    return PC


def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    '''

    '''
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    '''
    正确标记的道路像素 / (网络输出和真实道路像素之和)
    '''
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    Inter = torch.sum(((SR+GT)==2).int())
    Union = torch.sum(((SR+GT)>=1).int())
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    SR = SR.int()
    GT = GT.int()

    Inter = torch.sum(((SR+GT)==2).int())
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



