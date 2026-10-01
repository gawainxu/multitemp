#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:00:10 2021

"""

import json
import pickle
import numpy as np


def featureMerge(featureList, opt):
    
    head_features = []
    encoder_features = []
    labels = []
    print(opt.save_path_all)

    for featurePath in featureList:
        
        with open(featurePath, "rb") as f:
            head_feature, encoder_feature, labels_part = pickle.load(f)

        head_features = head_features + head_feature
        encoder_features = encoder_features + encoder_feature
        labels = labels + labels_part

    labels = np.squeeze(np.array(labels))
    
    with open(opt.save_path_all, 'wb') as f:
        pickle.dump((head_features, encoder_features, labels), f)
        
