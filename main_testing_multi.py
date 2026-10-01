import os
import sys
sys.path.append(BASE_PATH) 

import argparse
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import pickle
import copy
from itertools import chain
from scipy.spatial.distance import mahalanobis

from networks.resnet_big import SupConResNet, LinearClassifier

from util import  feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils  import sortFeatures
from dataUtil import get_test_datasets, get_outlier_datasets


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/")
    parser.add_argument("--model_path1", type=str, default=None)
    parser.add_argument("--model_path2", type=str, default=None)
    parser.add_argument("--ensembles", type=int, default=1)
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=20)
    
    parser.add_argument("--exemplar_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_1_128_512_1.0_0.5_0.1_256_train")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_1_128_512_1.0_0.5_0.1_256_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features/tinyimgnet_resnet_multi_trail_1_128_512_1.0_0.5_0.1_256_test_unknown")

    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--temp", type=str, default = 0.5)
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="train", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)

    parser.add_argument("--K", type=int, default=3)

    parser.add_argument("--auroc_save_path", type=str, default="./plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path

    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path

    return opt

def KNN_logits(testing_features, sorted_exemplar_features):

    testing_similarity_logits = []

    for idx, testing_feature in enumerate(testing_features):
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))

        testing_similarity_logits.append(similarity_logits)
    
    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits, axis=1)).T                         # normalization

    return testing_similarity_logits


def distances(stats, test_features, mode="other"):

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
                if np.linalg.cond(var) < 1/sys.float_info.epsilon:
                    var_rev = np.linalg.inv(var)
                    dis = mahalanobis(features, mu, var_rev)
                else:
                    dis = np.linalg.norm(features_normalized)
                    #var_rev = np.linalg.pinv(var)
                    #dis = mahalanobis(features, mu, var_rev)
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            diss.append(dis)
        
        dis_logits_out.append(np.min(np.array(diss))/np.sum(np.array(diss)))                   #  !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss))/np.sum(np.array(diss))) 
        dis_preds.append(np.argmin(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds


def KNN_classifier(testing_features, testing_labels, sorted_training_features):

    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits, axis=1)       

    acc = accuracy_plain(predictions, testing_labels)

    return prediction_logits, predictions, acc


def distance_classifier(testing_features, testing_labels, sorted_training_features):

    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds = distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)

    return np.array(dis_logits_in), np.array(dis_logits_out), np.array(dis_preds), acc


def layer_ratios(acc_known_dis, acc_known_dis1):

    layer_ratio1 = acc_known_dis / (acc_known_dis + acc_known_dis1)
    layer_ratio2 = acc_known_dis1 / (acc_known_dis + acc_known_dis1)
    return layer_ratio1, layer_ratio2


def sort_multihead(multihead_features):

    head1, head2, head3 = [], [], []
    for i, multi_features in enumerate(multihead_features):
        feature1, feature2, feature3 = multi_features
        head1.append(feature1.detach().cpu().numpy())
        head2.append(feature2.detach().cpu().numpy())
        head3.append(feature3.detach().cpu().numpy())

    return np.squeeze(np.array(head1)), np.squeeze(np.array(head2)), np.squeeze(np.array(head3))

def cat_examplars(sorted_exemplar_features1, sorted_exemplar_features2, sorted_exemplar_features3):

    sorted_exemplar_features = []
    for exemplar_features1, exemplar_features2, exemplar_features3 in zip(sorted_exemplar_features1, sorted_exemplar_features2, sorted_exemplar_features3):
        exemplar_features_c = [np.concatenate((f1, f2, f3), axis=0) for f1, f2, f3 in zip(exemplar_features1, exemplar_features2, exemplar_features3)]
        sorted_exemplar_features.append(exemplar_features_c)

    return sorted_exemplar_features


def cat_examplars2(sorted_exemplar_features1, sorted_exemplar_features2):

    sorted_exemplar_features = []
    for exemplar_features1, exemplar_features2 in zip(sorted_exemplar_features1, sorted_exemplar_features2):
        exemplar_features_c = [np.concatenate((f1, f2), axis=0) for f1, f2 in zip(exemplar_features1, exemplar_features2)]
        sorted_exemplar_features.append(exemplar_features_c)

    return sorted_exemplar_features


def normalize_scores(positive_scores, negative_scores):

    positive_max, positive_min = max(positive_scores), min(negative_scores)
    normalized_positive_scores = (positive_scores - positive_min) / (positive_max - positive_min)
    normalized_negative_scores = (negative_scores - positive_min) / (positive_max - positive_min)
    return normalized_positive_scores, normalized_negative_scores


def feature_classifier(opt):

    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_head, _, labels_examplar = pickle.load(f)         #
        features_head1, features_head2, features_head3 = sort_multihead(features_exemplar_head)
        sorted_features_examplar_head1 = sortFeatures(features_head1, labels_examplar, opt)
        sorted_features_examplar_head2 = sortFeatures(features_head2, labels_examplar, opt)
        sorted_features_examplar_head3 = sortFeatures(features_head3, labels_examplar, opt)

    with open(opt.testing_known_features_path, "rb") as f:
        features_testing_known_head, _, labels_testing_known = pickle.load(f)
        features_testing_known_head1, features_testing_known_head2, features_testing_known_head3 = sort_multihead(features_testing_known_head)

    features_testing_known_head1, labels_testing_known1 = down_sampling(features_testing_known_head1, opt.downsampling_ratio_known, labels_testing_known)
    prediction_logits_known1, predictions_known1, acc_known1 = KNN_classifier(features_testing_known_head1,
                                                                              labels_testing_known1,
                                                                              sorted_features_examplar_head1)
    prediction_logits_known_dis_in1, prediction_logits_known_dis_out1, predictions_known_dis1, acc_known_dis1 = distance_classifier(features_testing_known_head1,
                                                                                                                                    labels_testing_known1,
                                                                                                                                    sorted_features_examplar_head1)
    print("Distance Accuracy 1 is: ", acc_known_dis1)
    print("KNN Accuracy 1 is: ", acc_known1)

    features_testing_known_head2, labels_testing_known2 = down_sampling(features_testing_known_head2, opt.downsampling_ratio_known, labels_testing_known)
    prediction_logits_known2, predictions_known2, acc_known2 = KNN_classifier(features_testing_known_head2,
                                                                              labels_testing_known2,
                                                                              sorted_features_examplar_head2)
    prediction_logits_known_dis_in2, prediction_logits_known_dis_out2, predictions_known_dis2, acc_known_dis2 = distance_classifier(features_testing_known_head2,
                                                                                                                                    labels_testing_known2,
                                                                                                                                    sorted_features_examplar_head2)
    print("Distance Accuracy 2 is: ", acc_known_dis2)
    print("KNN Accuracy 2 is: ", acc_known2)

    features_testing_known_head3, labels_testing_known3 = down_sampling(features_testing_known_head3, opt.downsampling_ratio_known, labels_testing_known)
    prediction_logits_known3, predictions_known3, acc_known3 = KNN_classifier(features_testing_known_head3,
                                                                              labels_testing_known3,
                                                                              sorted_features_examplar_head3)
    prediction_logits_known_dis_in3, prediction_logits_known_dis_out3, predictions_known_dis3, acc_known_dis3 = distance_classifier(features_testing_known_head3,
                                                                                                                                    labels_testing_known3,
                                                                                                                                    sorted_features_examplar_head3)
    print("Distance Accuracy 3 is: ", acc_known_dis3)
    print("KNN Accuracy 3 is: ", acc_known3)

    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_head, _, labels_testing_unknown = pickle.load(f)
        features_testing_unknown_head1, features_testing_unknown_head2, features_testing_unknown_head3 = sort_multihead(features_testing_unknown_head)

    features_testing_unknown_head1, labels_testing_unknown1 = down_sampling(features_testing_unknown_head1,
                                                                            opt.downsampling_ratio_unknown,
                                                                            labels_testing_unknown)
    prediction_logits_unknown1, predictions_unknown1, _ = KNN_classifier(features_testing_unknown_head1,
                                                                         labels_testing_unknown1,
                                                                         sorted_features_examplar_head1)
    prediction_logits_unknown_dis_in1, prediction_logits_unknown_dis_out1, predictions_unknown_dis1, acc_unknown_dis1 = distance_classifier(
        features_testing_unknown_head1, labels_testing_unknown1, sorted_features_examplar_head1)

    features_testing_unknown_head2, labels_testing_unknown2 = down_sampling(features_testing_unknown_head2, opt.downsampling_ratio_known, labels_testing_unknown)
    prediction_logits_unknown2, predictions_unknown2, _ = KNN_classifier(features_testing_unknown_head2,
                                                                         labels_testing_unknown2,
                                                                         sorted_features_examplar_head2)
    prediction_logits_unknown_dis_in2, prediction_logits_unknown_dis_out2, predictions_unknown_dis2, acc_unknown_dis2 = distance_classifier(
        features_testing_unknown_head2, labels_testing_unknown2, sorted_features_examplar_head2)

    features_testing_unknown_head3, labels_testing_unknown3 = down_sampling(features_testing_unknown_head3, opt.downsampling_ratio_known, labels_testing_unknown)
    prediction_logits_unknown3, predictions_unknown3, _ = KNN_classifier(features_testing_unknown_head3,
                                                                         labels_testing_unknown3,
                                                                         sorted_features_examplar_head3)
    prediction_logits_unknown_dis_in3, prediction_logits_unknown_dis_out3, predictions_unknown_dis3, acc_unknown_dis3 = distance_classifier(
        features_testing_unknown_head3, labels_testing_unknown3, sorted_features_examplar_head3)

    
    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known1]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown1]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    norm_predictions_known1, norm_predictions_unknown1 = normalize_scores(prediction_logits_known1, prediction_logits_unknown1)
    norm_predictions_known2, norm_predictions_unknown2 = normalize_scores(prediction_logits_known2, prediction_logits_unknown2)
    norm_predictions_known3, norm_predictions_unknown3 = normalize_scores(prediction_logits_known3, prediction_logits_unknown3)
    prediction_logits_known = norm_predictions_known1 + norm_predictions_known2 #+ norm_predictions_known3
    prediction_logits_unknown = norm_predictions_unknown1 + norm_predictions_unknown2 #+ norm_predictions_unknown3
    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0)

    auroc_all = AUROC(labels_binary, probs_binary, opt)
    print("AUROC All is: ", auroc_all)

    probs_binary1 = np.concatenate((prediction_logits_known1, prediction_logits_unknown1), axis=0)
    auroc1 = AUROC(labels_binary, probs_binary1, opt)
    print("AUROC 1: ", auroc1)

    probs_binary2 = np.concatenate((prediction_logits_known2, prediction_logits_unknown2), axis=0)
    auroc2 = AUROC(labels_binary, probs_binary2, opt)
    print("AUROC 1: ", auroc2)

    probs_binary3 = np.concatenate((prediction_logits_known3, prediction_logits_unknown3), axis=0)
    auroc3 = AUROC(labels_binary, probs_binary3, opt)
    print("AUROC 3: ", auroc3)


    probs_binary_dis1 = np.concatenate((prediction_logits_known_dis_in1, prediction_logits_unknown_dis_in1), axis=0)
    auroc_dis1 = AUROC(labels_binary, probs_binary_dis1, opt)
    print("Dis AUROC 1 is: ", auroc_dis1)

    probs_binary_dis2 = np.concatenate((prediction_logits_known_dis_in2, prediction_logits_unknown_dis_in2), axis=0)
    auroc_dis2 = AUROC(labels_binary, probs_binary_dis2, opt)
    print("Dis AUROC 2 is: ", auroc_dis2)

    probs_binary_dis3 = np.concatenate((prediction_logits_known_dis_in3, prediction_logits_unknown_dis_in3), axis=0)
    auroc_dis3 = AUROC(labels_binary, probs_binary_dis3, opt)
    print("Dis AUROC 3 is: ", auroc_dis3)

    norm_prediction_logits_known_dis_in1, norm_prediction_logits_unknown_dis_in1 = normalize_scores(prediction_logits_known_dis_in1, prediction_logits_unknown_dis_in1)
    norm_prediction_logits_known_dis_in2, norm_prediction_logits_unknown_dis_in2 = normalize_scores(prediction_logits_known_dis_in2, prediction_logits_unknown_dis_in2)
    norm_prediction_logits_known_dis_in3, norm_prediction_logits_unknown_dis_in3 = normalize_scores(prediction_logits_known_dis_in3, prediction_logits_unknown_dis_in3)
    prediction_logits_known_dis_in = norm_prediction_logits_known_dis_in1 + norm_prediction_logits_known_dis_in2 #+ norm_prediction_logits_known_dis_in3
    prediction_logits_unknown_dis_in = norm_prediction_logits_unknown_dis_in1 + norm_prediction_logits_unknown_dis_in2 #+ norm_prediction_logits_unknown_dis_in3
    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0)

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)


    sorted_features_examplar_head = cat_examplars2(sorted_features_examplar_head1, sorted_features_examplar_head2)   # ,sorted_features_examplar_head2
    features_testing_known_head = np.concatenate(
        (features_testing_known_head1, features_testing_known_head2), axis=1)   # , features_testing_known_head3
    features_testing_unknown_head = np.concatenate(
        (features_testing_unknown_head1, features_testing_unknown_head2), axis=1)  # , features_testing_unknown_head3
    prediction_logits_known, predictions_known, acc_cat = KNN_classifier(features_testing_known_head,
                                                                            labels_testing_known1,
                                                                            sorted_features_examplar_head)
    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_head,
                                                                       labels_testing_unknown1,
                                                                       sorted_features_examplar_head)
    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0)
    print("KNN Accuracy cat is: ", acc_cat)

    auroc_cat = AUROC(labels_binary, probs_binary, opt)
    print("AUROC cat is: ", auroc_cat)

    prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
        features_testing_known_head,
        labels_testing_known1,
        sorted_features_examplar_head)
    prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, _ = distance_classifier(
        features_testing_unknown_head, labels_testing_unknown1, sorted_features_examplar_head)
    print("Dis Accuracy cat is: ", acc_known_dis)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0)
    auroc_cat_dis = AUROC(labels_binary, probs_binary_dis, opt)
    print("AUROC cat dis is: ", auroc_cat_dis)


    return auroc

        

if __name__ == "__main__":
    
    opt = parse_option()
    
    auroc = feature_classifier(opt)                        # oscr, acc_known
  
