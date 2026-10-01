#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
BASE_PATH = "/home/sysgen/Jiawen/SupContrast-master"
sys.path.append(BASE_PATH) 

import argparse

import torch
import numpy as np
import pickle
import copy
from scipy.spatial.distance import mahalanobis

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive

from util import  feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils  import sortFeatures
from dataUtil import get_test_datasets, get_outlier_datasets

from sklearn.neighbors import LocalOutlierFactor


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18",  choices=["resnet18", "resnet34", "vgg16", "preactresnet34", "simCNN"])
    parser.add_argument("--end", type=bool, default=False, help="if it is end to end training")
    parser.add_argument("--ensembles", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--feat_dim", type=int, default=128)
    
    parser.add_argument("--exemplar_features_path", type=str, default="/features/cifar10_resnet18_vanilia__SimCLR_0.8_0.2_0.05_trail_0_128_256_old_augmented_600_outliers_0_train")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/cifar10_resnet18_vanilia__SimCLR_0.8_0.2_0.05_trail_0_128_256_old_augmented_600_outliers_0_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features/cifar10_resnet18_vanilia__SimCLR_0.8_0.2_0.05_trail_0_128_256_old_augmented_600_outliers_0_test_unknown")

    parser.add_argument("--exemplar_features_path1", type=str, default=None)
    parser.add_argument("--testing_known_features_path1", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path1", type=str, default=None)

    parser.add_argument("--exemplar_features_path2", type=str, default=None)
    parser.add_argument("--testing_known_features_path2", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path2", type=str, default=None)

    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'multi_head'], help='choose method')
    parser.add_argument("--temp", type=str, default = 0.5)
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)

    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--LoF_K", type=int, default=5)
    parser.add_argument("--LoF_contamination", type=float, default=0.01)

    parser.add_argument("--auroc_save_path", type=str, default="/plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0)

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path
    opt.prediction_save_path = opt.exemplar_features_path.split("/")[-1]
    opt.prediction_save_path.replace("_train", "")
    opt.prediction_save_path = opt.prediction_save_path + "_predictions"
    opt.prediction_save_path = opt.main_dir + "/" + opt.prediction_save_path

    if opt.exemplar_features_path1 is not None:
        opt.exemplar_features_path1 = opt.main_dir + opt.exemplar_features_path1
    if opt.testing_known_features_path1 is not None:
        opt.testing_known_features_path1 = opt.main_dir + opt.testing_known_features_path1
    if opt.testing_unknown_features_path1 is not None:
        opt.testing_unknown_features_path1 = opt.main_dir + opt.testing_unknown_features_path1

    if opt.exemplar_features_path2 is not None:
        opt.exemplar_features_path2 = opt.main_dir + opt.exemplar_features_path2
    if opt.testing_known_features_path2 is not None:
        opt.testing_known_features_path2 = opt.main_dir + opt.testing_known_features_path2
    if opt.testing_unknown_features_path2 is not None:
        opt.testing_unknown_features_path2 = opt.main_dir + opt.testing_unknown_features_path2

    return opt


def set_loader(opt):
    # construct data loader
    test_dataset = get_test_datasets(opt)
    outlier_dataset = get_outlier_datasets(opt)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    outlier_loader = torch.utils.data.DataLoader(outlier_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)

    return test_loader, outlier_loader


def KNN_logits(testing_features, sorted_exemplar_features):

    testing_similarity_logits = []

    #testing_features = testing_features.astype(np.double)
    #testing_features = testing_features / np.linalg.norm(testing_features, axis=1)[:, np.newaxis]  ####

    for idx, testing_feature in enumerate(testing_features):
        #print(idx)
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            #training_features_c = training_features_c[::2]                                                          # TODO

            
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            #similarity_logits.append(np.sum(top_k_similarities))      #!!!!
            similarity_logits.append(top_k_similarities[-1])
            
            """
            training_features_c = training_features_c.astype(np.double)
            training_features_c = training_features_c / np.linalg.norm(training_features_c, axis=1)[:, np.newaxis]
            diff = training_features_c - testing_feature
            diff = diff.astype(np.double)
            similarities = np.linalg.norm((diff), axis=1)
            similarity_logits.append(np.min(similarities)) 
            """

        testing_similarity_logits.append(similarity_logits)
    
    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits, axis=1)).T                         # normalization, maybe not necessary???

    return testing_similarity_logits



def LoF(testing_features, sorted_exemplar_features, opt):
    
    scores = []
    for idx, testing_feature in enumerate(testing_features):
        
        clost_points = None
        clost_similarities = 0
        
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            if np.sum(similarities) > clost_similarities:
                clost_points = training_features_c
                clost_similarities = np.sum(similarities)
                
        # start LoF
        clf = LocalOutlierFactor(n_neighbors=opt.LoF_K, novelty=True, metric="cosine", contamination=opt.LoF_contamination)
        clf.fit(clost_points)
        testing_feature = np.expand_dims(testing_feature, axis=0)
        score = clf.decision_function(testing_feature)
        scores.append(score)
        
    return scores


def distances(stats, test_features, mode="normal"):

    dis_logits_out = []
    dis_logits_in = []
    dis_preds = []
    for features in test_features:
        diss = []
        for i, (mu, var) in enumerate(stats):
            #mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                #dis =  np.matmul(features_normalized, np.linalg.inv(var))
                #dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                #dis = dis[0][0]
                dis = mahalanobis(features, mu, np.linalg.inv(var))
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            diss.append(dis)
        
        dis_logits_out.append(np.min(np.array(diss))/np.sum(np.array(diss)))                   #  !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss))) 
        dis_preds.append(np.argmin(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds


def KNN_classifier(testing_features, testing_labels, sorted_training_features):

    print("Begin KNN Classifier!")
    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits, axis=1)
    #prediction_logits, predictions = -np.amin(testing_similarity_logits, axis=1), np.argmin(testing_similarity_logits, axis=1)       # minus here, larger score for inliers

    acc = accuracy_plain(predictions, testing_labels)
    print("KNN Accuracy is: ", acc)

    return prediction_logits, predictions, acc


def distance_classifier(testing_features, testing_labels, sorted_training_features):

    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds =  distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("Distance Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc


def sort_multiheadfeatures(multiheadfeatures):

    features_len = len(multiheadfeatures)
    sorted_features = []
    for i in range(features_len):
        f1, f2, f3 = multiheadfeatures[i]
        f1, f2, f3 = torch.squeeze(f1), torch.squeeze(f2), torch.squeeze(f3)
        feature_ensemble = torch.cat((f1,f2, f3))
        sorted_features.append(feature_ensemble.detach().numpy())

    return np.array(sorted_features)


def feature_classifier(opt):

    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_head, features_exemplar_backbone, labels_examplar = pickle.load(f)
        labels_examplar = np.squeeze(np.array(labels_examplar))
        if "multi_head" in opt.method:
            features_exemplar_head = sort_multiheadfeatures(features_exemplar_head)
        else:
            features_exemplar_head = np.squeeze(np.array(features_exemplar_head))
            features_exemplar_backbone = np.squeeze(np.array(features_exemplar_backbone))

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:
            features_exemplar_head1, _, labels_examplar1 = pickle.load(f)
            labels_examplar1 = np.squeeze(np.array(labels_examplar1))
        if "multi_head" in opt.method:
            features_exemplar_head1 = sort_multiheadfeatures(features_exemplar_head1)
        else:
            features_exemplar_head1 = np.squeeze(np.array(features_exemplar_head1))
            features_exemplar_head= np.concatenate((features_exemplar_head, features_exemplar_head1), axis=1)
            labels_examplar = np.concatenate((labels_examplar, labels_examplar1), axis=1)
    
    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:
            features_exemplar_head2, _, labels_examplar2 = pickle.load(f)
            labels_examplar2 = np.squeeze(np.array(labels_examplar2))
        if "multi_head" in opt.method:
            features_exemplar_head2 = sort_multiheadfeatures(features_exemplar_head2)
        else:
            features_exemplar_head2 = np.squeeze(np.array(features_exemplar_head2))
            features_exemplar_head = np.concatenate((features_exemplar_head, features_exemplar_head2), axis=1)
            labels_examplar = np.concatenate((labels_examplar, labels_examplar2), axis=1)

    sorted_features_exemplar_head = sortFeatures(features_exemplar_head, labels_examplar, opt)
    sorted_features_exemplar_backbone = sortFeatures(features_exemplar_backbone, labels_examplar, opt)

    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_head, features_testing_known_backbone, labels_testing_known = pickle.load(f)
            labels_testing_known = np.squeeze(np.array(labels_testing_known))
            if "multi_head" in opt.method:
                features_testing_known_head = sort_multiheadfeatures(features_testing_known_head)
            else:
                features_testing_known_head = np.squeeze(np.array(features_testing_known_head))
                features_testing_known_backbone = np.squeeze(np.array(features_testing_known_backbone))
    
    if opt.testing_known_features_path1 is not None:
        with open(opt.testing_known_features_path1, "rb") as f:
            features_testing_known_head1, _, _, labels_testing_known1 = pickle.load(f)
            labels_testing_known1 = np.squeeze(np.array(labels_testing_known1))
        if "multi_head" in opt.method:
            features_testing_known_head1 = sort_multiheadfeatures(features_testing_known_head1)
        else:
            features_testing_known_head1 = np.squeeze(np.array(features_testing_known_head1))
            features_testing_known_head1 = np.concatenate((features_testing_known_head, features_testing_known_head1), axis=1)
            labels_testing_known = np.concatenate((labels_testing_known, labels_testing_known1), axis=1)

    if opt.testing_known_features_path2 is not None:
        with open(opt.testing_known_features_path2, "rb") as f:
            features_testing_known_head2, _, labels_testing_known2 = pickle.load(f)
            labels_testing_known2 = np.squeeze(np.array(labels_testing_known2))
        if "multi_head" in opt.method:
            features_testing_known_head2 = sort_multiheadfeatures(features_testing_known_head2)
        else:
            features_testing_known_head2 = np.squeeze(np.array(features_testing_known_head2))
            features_testing_known_head = np.concatenate((features_testing_known_head, features_testing_known_head2), axis=1)
            labels_testing_known = np.concatenate((labels_testing_known, labels_testing_known2), axis=1)


    features_testing_known_backbone, labels_testing_known = down_sampling(features_testing_known_backbone, labels_testing_known, opt.downsampling_ratio_known)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_backbone, labels_testing_known, sorted_features_exemplar_backbone)
    prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(features_testing_known_backbone, labels_testing_known, sorted_features_exemplar_backbone)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_head, features_testing_unknown_backbone, labels_testing_unknown = pickle.load(f)
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))
    if "multi_head" in opt.method:
        features_testing_unknown_head = sort_multiheadfeatures(features_testing_unknown_head)
    else:
        features_testing_unknown_head = np.squeeze(np.array(features_testing_unknown_head))
        features_testing_unknown_backbone = np.squeeze(np.array(features_testing_unknown_backbone))
        

    if opt.testing_unknown_features_path1 is not None:
        with open(opt.testing_unknown_features_path1, "rb") as f:
            features_testing_unknown_head1, _, _, labels_testing_unknown1 = pickle.load(f)
            labels_testing_unknown1 = np.squeeze(np.array(labels_testing_unknown1))
        if "multi_head" in opt.method:
            features_testing_unknown_head1 = sort_multiheadfeatures(features_testing_unknown_head1)
        else:
            features_testing_unknown_head1 = np.squeeze(np.array(features_testing_unknown_head1))
            features_testing_unknown_head = np.concatenate((features_testing_known_head, features_testing_unknown_head1),axis=1)
            labels_testing_unknown = np.concatenate((labels_testing_unknown, labels_testing_unknown1), axis=1)
    
    if opt.testing_unknown_features_path2 is not None:
        with open(opt.testing_unknown_features_path2, "rb") as f:
            features_testing_unknown_head2, _, _, labels_testing_unknown2 = pickle.load(f)
            labels_testing_unknown2 = np.squeeze(np.array(labels_testing_unknown2))
        if "multi_head" in opt.method:
            features_testing_unknown_head2 = sort_multiheadfeatures(features_testing_unknown_head2)
        else:
            features_testing_unknown_head2 = np.squeeze(np.array(features_testing_unknown_head2))
            features_testing_unknown_head = np.concatenate((features_testing_unknown_head, features_testing_unknown_head2),axis=1)
            labels_testing_unknown = np.concatenate((labels_testing_unknown, labels_testing_unknown2), axis=1)

    features_testing_unknown_backbone, labels_testing_unknown = down_sampling(features_testing_unknown_backbone, labels_testing_unknown, opt.downsampling_ratio_unknown)
    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_backbone, labels_testing_unknown, sorted_features_exemplar_backbone)
    prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(features_testing_unknown_backbone, labels_testing_unknown, sorted_features_exemplar_backbone)
    
    knn_predictions = np.concatenate((predictions_known, predictions_unknown), axis=0)
    distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)

    with open(opt.prediction_save_path, "wb") as f:
        pickle.dump((knn_predictions, distance_predictions, labels_testing), f)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)
    #print("labels_binary", labels_binary)

    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0) 
    #with open("./prediction_logits_unknown_train_down", "wb") as f:
    #    pickle.dump(prediction_logits_unknown, f)

    # TODO visualize the scores !!!!!
    #plt.scatter(range(len(prediction_logits_known_dis_in)), prediction_logits_known_dis_in)
    #plt.savefig("./prediction_logits_known_dis_in.pdf")
    #plt.close("all")
    #plt.scatter(range(len(prediction_logits_unknown_dis_in)), prediction_logits_unknown_dis_in)
    #plt.savefig("./prediction_logits_unknown_dis_in.pdf")

    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC is: ", auroc)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0) 
    #print("probs_binary", probs_binary_dis)

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)
    
    # AUROC based on LoF
    #all_testing_features = np.concatenate((features_testing_known_backbone, features_testing_unknown_backbone), axis=0)
    #scores = LoF(all_testing_features, sorted_features_examplar_backbone, opt)
    #auroc = AUROC(labels_binary, scores, opt)
    #print("LoF AUROC is: ", auroc)

    # OSCR
    #oscr = OSCR(np.array(prediction_logits_known_dis_out), np.array(prediction_logits_unknown_dis_out), predictions_known, labels_testing_known)
    #print("OSCR is: ", oscr)

    #print("Acc Known: ", acc_known)

    return auroc             # oscr, acc_known

        
if __name__ == "__main__":
    
    opt = parse_option()
    
    auroc = feature_classifier(opt)                        # oscr, acc_known

    """
    1. use penultimate layer instead of head
    2. use ecudien distance 
    3. use kth distance instead of average distance
    4. feature normalization
    5. downsample training data
    """

    """
    pay attention to the samples at boundary
    """
