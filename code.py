import torch
import torch.nn as nn
import torch.optim 
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
from sklearn.metrics import accuracy_score, auc

import logging

import os

def create_empty_file(filename):
    with open(f"{filename}", "w") as my_empty_csv:
        pass
    

def append_to_file(line, filename):
    with open(f"{filename}", "a") as f:
        f.write(line)


def get_data(in_path='data', out_path='./imagenet-o'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    class vanilla_imagenet(torch.utils.data.Dataset):
        def __init__(self):
            self.data = datasets.ImageNet(in_path, split="val",
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  normalize]))

        def __getitem__(self, index):
            data, target = self.data[index]

            return data, target, index

        def __len__(self):
            return len(self.data)

    in_dataset = vanilla_imagenet()
    print(f'Inlier dataset length: {len(in_dataset)}')
    in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    class outlier_imagenet(torch.utils.data.Dataset):
        def __init__(self):
            self.data = datasets.ImageFolder(root=out_path, 
                                             transform=transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      normalize]))


        def __getitem__(self, index):
            data, target = self.data[index]

            return data, target, index

        def __len__(self):
            return len(self.data)

    out_dataset = outlier_imagenet()
    print(f'Outlier dataset length: {len(out_dataset)}')
    out_loader = torch.utils.data.DataLoader(out_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    return in_loader, out_loader

#######################################
#######################################
#######################################
#######################################
#######################################

def get_data_gridsearch(params, in_path='data', out_path='./imagenet-o', filename='EXPERIMENT-3-gridsearch-odin-with-gradient-log.txt'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    class vanilla_imagenet(torch.utils.data.Dataset):
        def __init__(self):
            self.data = datasets.ImageNet(in_path, split="val",
                                    transform=transforms.Compose([transforms.Resize(256),
                                                                  transforms.CenterCrop(224),
                                                                  transforms.ToTensor(),
                                                                  normalize]))

        def __getitem__(self, index):
            data, target = self.data[index]

            return data, target, index

        def __len__(self):
            return len(self.data)
    
    class outlier_imagenet(torch.utils.data.Dataset):
        def __init__(self):
            self.data = datasets.ImageFolder(root=out_path, 
                                             transform=transforms.Compose([transforms.Resize(256),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      normalize]))


        def __getitem__(self, index):
            data, target = self.data[index]

            return data, target, index

        def __len__(self):
            return len(self.data)
        
    in_dataset = vanilla_imagenet()
    out_dataset = outlier_imagenet()
    
    print(f'Inlier dataset total length: {len(in_dataset)}')
    print(f'Outlier dataset total length: {len(out_dataset)}')
    
    idx_in = list(range(len(in_dataset)))
    np.random.shuffle(idx_in)      
    test_idx_in = idx_in[ : int(params['test_ratio_in'] * len(idx_in))]       
    gridsearch_idx_in = idx_in[int(params['test_ratio_in'] * len(idx_in)) : ]
    test_sampler_in = torch.utils.data.sampler.SubsetRandomSampler(test_idx_in)    
    gridsearch_sampler_in = torch.utils.data.sampler.SubsetRandomSampler(gridsearch_idx_in) 
    
    in_loader_test = torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=test_sampler_in)
    in_loader_gridsearch = torch.utils.data.DataLoader(in_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=gridsearch_sampler_in)
    
    idx_out = list(range(len(out_dataset)))
    np.random.shuffle(idx_out)          
    test_idx_out = idx_out[ : int(params['test_ratio_out'] * len(idx_out))]
    gridsearch_idx_out = idx_out[int(params['test_ratio_out'] * len(idx_out)) : ]
    test_sampler_out = torch.utils.data.sampler.SubsetRandomSampler(test_idx_out)    
    gridsearch_sampler_out = torch.utils.data.sampler.SubsetRandomSampler(gridsearch_idx_out) 
    
    out_loader_test = torch.utils.data.DataLoader(out_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=test_sampler_out)
    out_loader_gridsearch = torch.utils.data.DataLoader(out_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, sampler=gridsearch_sampler_out)
    
    np.savetxt('test_idx_in.txt', np.array(test_idx_in).astype(int), fmt='%i')
    np.savetxt(f'test_idx_out_{out_path[2:]}.txt', np.array(test_idx_out).astype(int), fmt='%i')
    
    print(f'Inlier dataset test length: {len(test_idx_in)}')
    print(f'Outlier dataset test length: {len(test_idx_out)}')
    
    print(f'Inlier dataset gridsearch length: {len(gridsearch_idx_in)}')
    print(f'Outlier dataset gridsearch length: {len(gridsearch_idx_out)}')
    
    print('Validation:')
    print(f'inlier test first 10 indices: {test_idx_in[:10]}')
    print(f'outlier test first 10 indices: {test_idx_out[:10]}')
    print(f'inlier gridsearch first 10 indices: {gridsearch_idx_in[:10]}')
    print(f'outlier gridsearch first 10 indices: {gridsearch_idx_out[:10]}')
    
    create_empty_file(filename)
    
    append_to_file('#' * 100 + '\n', filename)
    
    append_to_file(f'Inlier dataset total length: {len(in_dataset)}' + '\n', filename)
    append_to_file(f'Outlier dataset total length: {len(out_dataset)}' + '\n', filename)
    append_to_file(f'Inlier dataset test length: {len(test_idx_in)}' + '\n', filename)
    append_to_file(f'Outlier dataset test length: {len(test_idx_out)}' + '\n', filename)
    
    append_to_file(f'Inlier dataset gridsearch length: {len(gridsearch_idx_in)}' + '\n', filename)
    append_to_file(f'Outlier dataset gridsearch length: {len(gridsearch_idx_out)}' + '\n', filename)
    
    append_to_file('#' * 100 + '\n', filename)
    
    append_to_file('Validation:' + '\n', filename)
    append_to_file(f'inlier test first 10 indices: {test_idx_in[:10]}' + '\n', filename)
    append_to_file(f'outlier test first 10 indices: {test_idx_out[:10]}' + '\n', filename)
    append_to_file(f'inlier gridsearch first 10 indices: {gridsearch_idx_in[:10]}' + '\n', filename)
    append_to_file(f'outlier gridsearch first 10 indices: {gridsearch_idx_out[:10]}' + '\n', filename)
    
    return in_loader_test, out_loader_test, in_loader_gridsearch, out_loader_gridsearch


#######################################
#######################################
#######################################
#######################################
#######################################

def vanilla_gradient(data_loader, net, device, silent=False):
    criterion = nn.CrossEntropyLoss()
    all_norms = []
    for cur_id, test_data in enumerate(data_loader):
        if not silent and cur_id % 5000 == 0:
            print(f'current iter: {cur_id}')

        inputs, actual_val, i = test_data

        net.zero_grad()
        predicted_val = net(torch.autograd.Variable(inputs.to(device)))
        max_score, idx = torch.max(predicted_val, dim=1)
        labels = torch.autograd.Variable(idx).to(device)
        loss = criterion(predicted_val, labels)
        loss.backward()

        full_norm = 0
        for name, param in net.named_parameters():
            cur_grad = param.grad.view(-1)
            #print(name, cur_grad.size())
            full_norm += (torch.norm(cur_grad.cpu()).item()) ** 2
        all_norms.append(full_norm ** (0.5))

    all_norms = np.array(all_norms)
    
    return all_norms

#######################################
#######################################
#######################################
#######################################
#######################################

def baseline(data_loader, net, device, silent=False):
    all_norms = []
    softmax = torch.nn.Softmax(dim=1)
    for cur_id, test_data in enumerate(data_loader):
        if not silent and cur_id % 5000 == 0:
            print(f'current iter: {cur_id}')

        inputs, actual_val, i = test_data
        
        predicted_val = net(torch.autograd.Variable(inputs.to(device)))
        proba = softmax(predicted_val)
        max_score, idx = torch.max(proba, 1)
        
        all_norms.append(-max_score.cpu().item())

    all_norms = np.array(all_norms)
    
    return all_norms

#######################################
#######################################
#######################################
#######################################
#######################################


def max_logit(data_loader, net, device, silent=False):
    all_norms = []
    softmax = torch.nn.Softmax(dim=1)
    for cur_id, test_data in enumerate(data_loader):
        if not silent and cur_id % 5000 == 0:
            print(f'current iter: {cur_id}')

        inputs, actual_val, i = test_data
        
        predicted_val = net(torch.autograd.Variable(inputs.to(device)))
        #proba = softmax(predicted_val)
        max_score, idx = torch.max(predicted_val, 1)
        
        all_norms.append(-max_score.cpu().item())

    all_norms = np.array(all_norms)
    
    return all_norms

#######################################
#######################################
#######################################
#######################################
#######################################

def odin(eps, temper, data_loader, net, device, silent=False):
    all_norms = []
    softmax = torch.nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    
    for i, test_data in enumerate(data_loader):
        if not silent and i % 5000 == 0:
            print(f'current iter: {i}')

        inputs, actual_val, i = test_data
        inputs = torch.autograd.Variable(inputs.to(device), requires_grad=True)
        outputs = net(inputs)
        outputs /= temper
        max_score, idx1 = torch.max(outputs, 1)
        loss = criterion(outputs, idx1)
        loss.backward()
        
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient[0][0] = (gradient[0][0]) / (0.229)
        gradient[0][1] = (gradient[0][1]) / (0.224)
        gradient[0][2] = (gradient[0][2]) / (0.225)
        temp_inputs = inputs.data - eps * gradient
        outputs = net(temp_inputs)
        outputs /= temper
        
        probas = softmax(outputs)
        
        max_score, idx = torch.max(probas, 1)
        
        all_norms.append(-max_score.cpu().item())

    all_norms = np.array(all_norms)
    
    return all_norms


def gridsearch_odin(in_loader, out_loader, net, device, filename='gridsearch-odin-log.txt', information='resnet + imagenet-o'):
    eps_list = np.linspace(0, 0.004, 21)
    temper_list = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])

    best_score = -1.0
    best_params = (0.0, 1.0)
    
    append_to_file('#' * 100 + '\n', filename)
    append_to_file(information + '\n', filename)
    append_to_file('#' * 100 + '\n', filename)

    for eps in eps_list:
        for temper in temper_list:
            in_norms = odin(eps, temper, in_loader, net, device, silent=True)
            out_norms = odin(eps, temper, out_loader, net, device, silent=True)

            thresholds = np.linspace(np.min(in_norms), np.max(out_norms), 10000)
            roc_auc, fpr_at_95 = my_roc_auc_score(in_norms, out_norms, thresholds, graph=False)

            if roc_auc >= best_score:
                best_score = roc_auc
                best_params = (eps, temper)


            append_to_file(f'eps = {eps}, T = {temper}: roc-auc={roc_auc}, fpr at 95% tpr={fpr_at_95}'+'\n', filename)

    append_to_file(f'best score = {best_score} is when eps = {best_params[0]}, T = {best_params[1]}'+'\n', filename)
                
    return best_params


#######################################
#######################################
#######################################
#######################################
#######################################

def bad_layer(name, banned_layers):
    return (name in banned_layers)


def odin_with_gradient(eps, temper, banned_layers, data_loader, net, device, silent=False, stop=1000000):
    all_norms = []
    softmax = torch.nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    for i, test_data in enumerate(data_loader):
        if not silent and i % 5000 == 0:
            print(f'current iter: {i}')
        if i == stop:
            break

        inputs, actual_val, i = test_data
        inputs = torch.autograd.Variable(inputs.to(device), requires_grad=True)
        outputs = net(inputs)
        outputs /= temper
        max_score, idx1 = torch.max(outputs, 1)
        loss = criterion(outputs, idx1)
        loss.backward()
        
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient[0][0] = (gradient[0][0]) / (0.229)
        gradient[0][1] = (gradient[0][1]) / (0.224)
        gradient[0][2] = (gradient[0][2]) / (0.225)
        temp_inputs = inputs.data - eps * gradient
        
        temp_inputs = torch.autograd.Variable(temp_inputs.to(device), requires_grad=True)
        
        net.zero_grad()
        
        outputs = net(temp_inputs)
        outputs /= temper
        
        max_score, idx = torch.max(outputs, 1)
        
        loss = criterion(outputs, idx)
        # loss = criterion(outputs, idx1)
        loss.backward()
        
        #full_norm = (torch.norm(temp_inputs.grad.view(-1).cpu()).item()) ** 2
        #print(full_norm)
        full_norm = 0
        for name, param in net.named_parameters():
            if bad_layer(name, banned_layers):
                continue
            cur_grad = param.grad.view(-1)
            #print(name, cur_grad.size())
            full_norm += (torch.norm(cur_grad.cpu()).item()) ** 2
        all_norms.append(full_norm ** (0.5))

    all_norms = np.array(all_norms)
    
    return all_norms


def gridsearch_odin_with_gradient(in_loader, out_loader, params, net, device, filename='gridsearch-odin-with-gradient-log.txt', information='resnet + imagenet-o'):
#     eps_list = np.linspace(0, 0.004, 21)
#     temper_list = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])

    best_score = -1.0
    best_params = (0.0, 1.0)
    
    append_to_file('#' * 100 + '\n', filename)
    append_to_file(information + '\n', filename)
    append_to_file('#' * 100 + '\n', filename)
    
    for eps in params['eps_list']:
        for temper in params['temper_list']:
            in_norms = odin_with_gradient(eps, temper, params['banned_layers'], in_loader, net, device, silent=True)
            out_norms = odin_with_gradient(eps, temper, params['banned_layers'], out_loader, net, device, silent=True)

            thresholds = np.linspace(np.min(in_norms), np.max(out_norms), 10000)
            roc_auc, fpr_at_95 = my_roc_auc_score(in_norms, out_norms, thresholds, graph=False)

            if roc_auc >= best_score:
                best_score = roc_auc
                best_params = (eps, temper)


            append_to_file(f'eps = {eps}, T = {temper}: roc-auc={roc_auc}, fpr at 95% tpr={fpr_at_95}'+'\n', filename)

    append_to_file(f'best score = {best_score} is when eps = {best_params[0]}, T = {best_params[1]}'+'\n', filename)
                
    return best_params

#######################################
#######################################
#######################################
#######################################
#######################################

def compare_histograms(in_norms, out_norms, in_name='imagenet', out_name='imagenet-o', filename='gridsearch-odin-with-gradient-log.txt'):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=in_norms, histnorm='probability', name=in_name))
    fig.add_trace(go.Histogram(x=out_norms, histnorm='probability', name=out_name))

    fig.update_layout(barmode='overlay', title=f'{in_name} VS {out_name}')
    fig.update_traces(opacity=0.75)
    
    fig.write_html(f"{filename[:-4]}.html")

    fig.show()

def check_accuracy(in_loader, net, device):
    net.eval()
    correct = 0.0
    total = 0.0
    for test_data in in_loader:
        total += 1
        inputs, actual_val, i = test_data 
        predicted_val = net(inputs.to(device))

        predicted_val = predicted_val.cpu().data

        max_score, idx = torch.max(predicted_val, 1)

        correct += (idx == actual_val).sum()

    print("Classifier Accuracy: ", correct/total * 100)


def get_stats(_all_norms, information='inliers:', filename='stats.txt'):
    print(f'min = {np.min(_all_norms)}')
    print(f'max = {np.max(_all_norms)}')
    print(f'median = {np.median(_all_norms)}')
    print(f'mean = {np.mean(_all_norms)}')
    print(f'std = {np.std(_all_norms)}')
    
    append_to_file('#' * 100 + '\n', filename)
    append_to_file(information + '\n', filename)
    append_to_file(f'min = {np.min(_all_norms)}'+ '\n', filename)
    append_to_file(f'max = {np.max(_all_norms)}'+ '\n', filename)
    append_to_file(f'median = {np.median(_all_norms)}'+ '\n', filename)
    append_to_file(f'mean = {np.mean(_all_norms)}'+ '\n', filename)
    append_to_file(f'std = {np.std(_all_norms)}'+ '\n', filename)
    append_to_file('#' * 100 + '\n', filename)
    
    
def OOD_metrics(all_norms_in, all_norms_out, threshold):
    # True class for outliers, False class for inliers
    TP = np.sum(all_norms_out >= threshold)
    
    TN = np.sum(all_norms_in < threshold)
    
    FN = np.sum(all_norms_out < threshold)
    
    FP = np.sum(all_norms_in >= threshold)
    
    confusion = np.array([[TP, FP], [FN, TN]])
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return confusion, f1
    
def my_roc_auc_score(all_norms_in, all_norms_out, thresholds, title='roc-curve', graph=True):
    x = []
    y = []
    
    fpr_at_95 = []
    
    for cur_threshold in thresholds:
        TP = np.sum(all_norms_out >= cur_threshold)
        TN = np.sum(all_norms_in < cur_threshold)
        FN = np.sum(all_norms_out < cur_threshold)
        FP = np.sum(all_norms_in >= cur_threshold)
        
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        
        if TPR >= 0.945 and TPR <= 0.955:
            fpr_at_95.append(FPR)
        x.append(FPR)
        y.append(TPR)
        
    x = np.array(x)
    y = np.array(y)
    fpr_at_95 = np.array(fpr_at_95)
    roc_auc_score = auc(x, y)
    
    if graph:
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', text=thresholds))

        fig.update_layout(title=title, width=500, height=500, xaxis_title="FPR", yaxis_title="TPR")
        fig.update_layout(yaxis=dict(range=[0, 1.1]))
        fig.update_layout(xaxis=dict(range=[0, 1.1]))
        

        fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color='black',),
                xref='x',
                yref='y'
        )

        fig.show()
        
    return roc_auc_score, np.mean(fpr_at_95)

def my_pr_auc_score(all_norms_in, all_norms_out, thresholds, mode='pos-out', title='precision-recall-curve', graph=True):
    x = []
    y = []
    
    for cur_threshold in thresholds:
        if mode == 'pos-out':
            TP = np.sum(all_norms_out >= cur_threshold)
            TN = np.sum(all_norms_in < cur_threshold)
            FN = np.sum(all_norms_out < cur_threshold)
            FP = np.sum(all_norms_in >= cur_threshold)
        else:
            TP = np.sum(all_norms_in < cur_threshold)
            TN = np.sum(all_norms_out >= cur_threshold)
            FN = np.sum(all_norms_in >= cur_threshold)
            FP = np.sum(all_norms_out < cur_threshold)
        
        if FN == 0:
            RECALL = 1.0
        else:
            RECALL = TP / (TP + FN)
        if FP == 0:
            PRECISION = 1.0
        else:
            PRECISION = TP / (TP + FP)
            
        x.append(RECALL)
        y.append(PRECISION)
        
    x = np.array(x)
    y = np.array(y)
    pr_auc_score = auc(x, y)
    
    if graph:
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', text=thresholds))

        fig.update_layout(title=title, width=500, height=500, xaxis_title="RECALL", yaxis_title="PRECISION")
        fig.update_layout(yaxis=dict(range=[0, 1.1]))
        fig.update_layout(xaxis=dict(range=[0, 1.1]))

        fig.show()
        
    return pr_auc_score


def calc_all_metrics(in_norms, out_norms, information='metrics:', filename='stats.txt'):
    thresholds = np.linspace(np.min(in_norms), np.max(in_norms), 10000)
    
    roc_auc, fpr_at_95 = my_roc_auc_score(in_norms, out_norms, thresholds, graph=False)
    print(f'roc-auc-score: {roc_auc}')
    print(f'fpr at 95% tpr: {fpr_at_95}')
    
    score1 = my_pr_auc_score(in_norms, out_norms, thresholds, graph=False)
    print(f'precision-recall-auc-score (positive-outlier, negative-inlier): {score1}')
    
    score2 = my_pr_auc_score(in_norms, out_norms, thresholds, mode='pos-in', graph=False)
    print(f'precision-recall-auc-score (positive-inlier, negative-outlier): {score2}')
    
    append_to_file('#' * 100 + '\n', filename)
    append_to_file(information + '\n', filename)
    append_to_file(f'roc-auc-score: {roc_auc}' + '\n', filename)
    append_to_file(f'fpr at 95% tpr: {fpr_at_95}' + '\n', filename)
    append_to_file(f'precision-recall-auc-score (positive-outlier, negative-inlier): {score1}' + '\n', filename)
    append_to_file(f'precision-recall-auc-score (positive-inlier, negative-outlier): {score2}' + '\n', filename)
    append_to_file('#' * 100 + '\n', filename)
    

#######################################
#######################################
#######################################
#######################################
#######################################
    
    
def freq_hist_bins_by_median(all_norms, mask, title, points_in_bin = 800):
    sorted_ids = np.argsort(all_norms)
    
    num_of_bins = len(all_norms) // points_in_bin
    
    if len(all_norms) % points_in_bin != 0:
        num_of_bins += 1

    hist_x = []
    
    number_of_subsamples = 100
    hist_median = []
    hist_percentile = []
    min_norm_by_bin = []
    
    for i in range(1, num_of_bins + 1):
        cur_bin_ids = sorted_ids[(i - 1) * points_in_bin: i * points_in_bin]
        
        cur_bin_ids_RAY = sorted_ids[(i - 1) * points_in_bin:]
        
        min_norm_by_bin.append(np.min(all_norms[cur_bin_ids]))
        
        hist_x.append(np.median(all_norms[cur_bin_ids]))
        
        accuracies = []
        for _ in range(number_of_subsamples):
            cur_subsample = np.random.choice(cur_bin_ids_RAY, size=len(cur_bin_ids_RAY))

            accuracy = np.sum(mask[cur_subsample]) / len(cur_bin_ids_RAY)
            accuracies.append(accuracy)

        accuracies = np.array(accuracies)
        hist_median.append(np.median(accuracies))
        hist_percentile.append(np.percentile(accuracies, 1))

    hist_x = np.array(hist_x)
    hist_median = np.array(hist_median)
    hist_percentile = np.array(hist_percentile)
    
    min_norm_by_bin = np.array(min_norm_by_bin)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_x, y=hist_median, mode='markers', text=np.arange(len(hist_x)), name='mean'))
    fig.add_trace(go.Scatter(x=hist_x, y=hist_percentile, mode='markers', text=np.arange(len(hist_x)), name='LCB'))
    
    fig.add_shape(type='line',
                x0=0,
                y0=0.1,
                x1=110,
                y1=0.1,
                line=dict(color='black',),
                xref='x',
                yref='y'
    )
    
    fig.update_layout(title=title)

    fig.show()
    
    return min_norm_by_bin

def save_and_bootstrap(exp_name, all_norms_in, all_norms_out, bootstrap_num=1000):
    path = f'bootstrap_results/{exp_name}'
    os.mkdir(path)
    
    np.savetxt(f'{path}/in_{exp_name}.csv', all_norms_in, delimiter=',')
    np.savetxt(f'{path}/out_{exp_name}.csv', all_norms_out, delimiter=',')
    
    all_roc_auc = []
    all_fpr = []
    all_pr_auc_pos = []
    all_pr_auc_neg = []
    
    thresholds = np.linspace(np.min(all_norms_in), np.max(all_norms_out), 10000)
    
    for _ in range(bootstrap_num):

        ids_in = np.random.choice(np.arange(len(all_norms_in)), size=len(all_norms_in))
        ids_out = np.random.choice(np.arange(len(all_norms_out)), size=len(all_norms_out))
        
        roc_auc, fpr = my_roc_auc_score(all_norms_in[ids_in], all_norms_out[ids_out], thresholds, title='', graph=False)
        pr_auc_pos = my_pr_auc_score(all_norms_in[ids_in], all_norms_out[ids_out], thresholds, mode='pos-out', title='', graph=False)
        pr_auc_neg = my_pr_auc_score(all_norms_in[ids_in], all_norms_out[ids_out], thresholds, mode='pos-neg', title='', graph=False)
        
        all_roc_auc.append(roc_auc)
        all_fpr.append(fpr)
        all_pr_auc_pos.append(pr_auc_pos)
        all_pr_auc_neg.append(pr_auc_neg)
        
        if _ % 10 == 0:
            print(f'bootstrap iter {_}! {len(all_norms_in[ids_in])}, {len(all_norms_out[ids_out])}')
            print(f'{roc_auc}, {fpr}, {pr_auc_pos}, {pr_auc_neg}')
        
    all_roc_auc = np.array(all_roc_auc)
    all_fpr = np.array(all_fpr)
    all_pr_auc_pos = np.array(all_pr_auc_pos)
    all_pr_auc_neg = np.array(all_pr_auc_neg)
    
    filename = f'{path}/res_{exp_name}.txt'
    
    create_empty_file(filename)
    
    append_to_file(f'roc-auc: {np.mean(all_roc_auc)} ± {np.std(all_roc_auc)}\n', filename)
    append_to_file(f'fpr: {np.mean(all_fpr)} ± {np.std(all_fpr)}\n', filename)
    append_to_file(f'pr_auc_pos: {np.mean(all_pr_auc_pos)} ± {np.std(all_pr_auc_pos)}\n', filename)
    append_to_file(f'pr_auc_neg: {np.mean(all_pr_auc_neg)} ± {np.std(all_pr_auc_neg)}\n', filename)
    
    print(f'roc-auc: {np.mean(all_roc_auc)} ± {np.std(all_roc_auc)}')
    print(f'fpr: {np.mean(all_fpr)} ± {np.std(all_fpr)}')
    print(f'pr_auc_pos: {np.mean(all_pr_auc_pos)} ± {np.std(all_pr_auc_pos)}')
    print(f'pr_auc_neg: {np.mean(all_pr_auc_neg)} ± {np.std(all_pr_auc_neg)}')