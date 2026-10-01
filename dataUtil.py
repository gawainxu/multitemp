#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:42:16 2021

"""


classMap = {0: "apples", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
            5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottles",
            10: "bowls", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
            15: "camel", 16: "cans", 17: "castle", 18: "caterpillar", 19: "cattle",
            20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
            25: "couch", 26: "crab", 27: "crocodile", 28: "cups", 29: "dinosaur",
            30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
            35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 
            40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
            45: "lobster", 46: "man", 47: "maple", 48: "motorcycle", 49: "mountain",
            50: "mouse", 51: "mushrooms", 52: "oak", 53: "oranges", 54: "orchids", 
            55: "otter", 56: "palm", 57: "pears", 58: "pickup_truck", 59: "pine",
            60: "plain", 61: "plates", 62: "poppies", 63: "porcupine", 64: "possum",
            65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
            70: "roses", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
            75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
            80: "squirrel", 81: "streetcar", 82: "sunflowers", 83: "pepper", 84: "table", 
            85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
            90: "train", 91: "trout", 92: "tulips", 93: "turtle", 94: "wardrobe",
            95: "whale", 96: "willow", 97: "wolf", 98: "woman", 99: "worm"}

classMap = {v : k for k, v in classMap.items()}

superClasses = [["beaver", "dolphin", "otter", "seal", "whale"],
                ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                ["orchids", "poppies", "roses", "sunflowers", "tulips"],
                ["bottles", "bowls", "cans", "cups", "plates"],
                ["apples", "mushrooms", "oranges", "pears", "peppers"],
                ["clock", "keyboard", "lamp", "telephone", "television"],
                ["bed", "chair", "couch", "table", "wardrobe"],
                ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                ["bear", "leopard", "lion", "tiger", "wolf"],
                ["bridge", "castle", "house", "road", "skyscraper"],
                ["cloud", "forest", "mountain", "plain", "sea"],
                ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                ["fox", "porcupine", "possum", "raccoon", "skunk"],
                ["crab", "lobster", "snail", "spider", "worm"],
                ["baby", "boy", "girl", "man", "woman"],
                ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                ["maple", "oak", "palm", "pine", "willow"],
                ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]]


osr_splits_inliers = {
    # Training classes
    'mnist': [
        [2, 4, 5, 9, 8, 3],
        [3, 2, 6, 9, 4, 0],
        [5, 8, 3, 2, 4, 6],
        [3, 7, 8, 4, 0, 5],
        [6, 3, 4, 9, 8, 2]
    ],
    'svhn': [
        [5, 3, 7, 2, 8, 6],
        [3, 8, 7, 6, 2, 5],
        [8, 9, 4, 7, 2, 1],
        [3, 8, 2, 5, 0, 6],
        [4, 9, 2, 7, 1, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ],

    'cifar10': [
        [0, 6, 4, 9, 1, 7],
        [7, 6, 4, 9, 0, 1],
        [1, 5, 7, 3, 9, 4],
        [8, 6, 1, 9, 0, 7],
        [2, 4, 1, 7, 9, 6],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [2, 3, 5, 8]
    ],

    'cifar-10-100-10': [
        [4, 7, 9, 1],
        [6, 7, 1, 9],
        [9, 6, 1, 7],
        [6, 4, 9, 1],
        [1, 0, 9, 8]
    ],

    'cifar-10-100-50': [
        [4, 7, 9, 1],
        [6, 7, 1, 9],
        [9, 6, 1, 7],
        [6, 4, 9, 1],
        [1, 0, 9, 8]
    ],

    'tinyimgnet': [
        [108, 147, 17, 58, 193, 123, 72, 144, 75, 167, 134, 14, 81, 171, 44, 197, 152, 66, 1, 133],
        [198, 161, 91, 59, 57, 134, 61, 184, 90, 35, 29, 23, 199, 38, 133, 19, 186, 18, 85, 67],
        [177, 0, 119, 26, 78, 80, 191, 46, 134, 92, 31, 152, 27, 60, 114, 50, 51, 133, 162, 93],
        [98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190],
        [95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170, 102, 177, 108, 169],
        [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  20,  21,  22,  24,  25,  26,  27,  28, 30,  31,  32,  33,  34,  36,  37,  39,  40,  41,  42,  43,  44,
         45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58, 60,  62,  63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74, 75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,
        ]
    ],
    
    "cub": [[150, 70, 34, 178, 199, 131, 129, 147, 134, 11, 26, 93, 95, 121, 123, 99, 149, 167,
            18, 31, 69, 198, 116, 158, 126, 17, 5, 179, 111, 163, 184, 81, 174, 42, 53, 89, 77,
            55, 23, 48, 43, 44, 56, 28, 193, 143, 0, 176, 84, 15, 38, 154, 141, 190, 172, 124,
            189, 19, 80, 157, 12, 9, 79, 30, 94, 67, 197, 97, 168, 137, 119, 76, 98, 88, 40, 106,
            171, 87, 166, 186, 27, 51, 144, 135, 161, 64, 177, 7, 146, 61, 50, 162, 133, 82, 39,
            74, 72, 91, 196, 136]], 
            
    "aircraft": [[0, 1, 2, 3, 4, 5, 10, 11, 14, 16, 17, 19, 21, 22, 23, 24, 27, 28, 29, 30, 33,
                  36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 52, 53, 56, 57, 58, 63, 64, 65,
                  66, 67, 71, 73, 76, 77, 79, 92, 95, 99]],

    "stanfordcars": [[1, 11, 25, 38, 46, 50, 53, 75, 84, 100, 105, 117, 123, 129, 133, 134, 135, 136,
                      137, 138, 140, 144, 145, 146, 147, 149, 150, 151, 153, 160, 161, 162, 163, 164,
                      167, 168, 169, 174, 175, 180, 185, 186, 187, 192, 193, 0, 81, 97, 104, 122, 139,
                      141, 142, 143, 148, 152, 154, 155, 156, 157, 158, 159, 165, 166, 170, 171, 172,
                      173, 176, 177, 181, 184, 188, 191, 194, 195, 2, 7, 9, 16, 20, 26, 28, 44, 54, 95,
                      98, 102, 127, 178, 182, 22, 41, 82, 93, 112, 125, 189]],
                  
    "cifar100_marco": [[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
                       [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96],
                       [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97],
                       [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98],
                       [0,1,5,6,10,11,15,16,20,21,25,26,30,31,35,36,40,41,45,46,50,51,55,56,60,61,65,
                        66,70,71,75,76,80,81,85,86,90,91,95,96],
                       [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27, 30, 31, 32,
                        35, 36, 37, 40, 41, 42, 45, 46, 47, 50, 51, 52, 55, 56, 57, 60, 61, 62, 65, 66,
                        67, 70, 71, 72, 75, 76, 77, 80, 81, 82, 85, 86, 87, 90, 91, 92, 95, 96, 97],
                       [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27,
                        28, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 52, 53,
                        55, 56, 57, 58, 60, 61, 62, 63, 65, 66, 67, 68, 70, 71, 72, 73, 75, 76, 77, 78, 80,
                        81, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 95, 96, 97, 98],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       list(range(100))
                       ],

    "cifar100": [list(range(10))]
                       #[[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
                       #[0, 1, 5, 6, 10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40, 41, 45, 46],
                       #[4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
                       #[3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
                       #[4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 3, 8, 13, 18, 23, 28, 33, 38, 43, 48]],
}


osr_splits_outliers = {
    # testing classes
    "mnist": [[0, 1, 6, 7],
              [1, 5, 7, 8],
              [0, 1, 7, 9],
              [1, 2, 6, 9],
              [0, 1, 5, 7]],

    "svhn": [[0, 1, 4, 9],
             [0, 1, 4, 9],
             [0, 3, 5, 6],
             [1, 4, 7, 9],
             [3, 5, 6, 8]],

    "cifar10": [[2, 3, 5, 8],
                [2, 3, 5, 8],
                [0, 2, 6, 8],
                [2, 3, 4, 5],
                [0, 3, 5, 8]],

    'cifar-10-100-10': [
        [30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],

    'cifar-10-100-50': [
        [27, 94, 29, 77, 88, 26, 69, 48, 75, 5, 59, 93, 39, 57, 45, 40, 78, 20, 98, 47, 66, 70, 91, 76, 41, 83, 99, 32, 53, 72, 2, 95, 21, 73, 84, 68, 35, 11, 55, 60, 30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [65, 97, 86, 24, 45, 67, 2, 3, 91, 98, 79, 29, 62, 82, 33, 76, 0, 35, 5, 16, 54, 11, 99, 52, 85, 1, 25, 66, 28, 84, 23, 56, 75, 46, 21, 72, 55, 68, 8, 69, 41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [20, 83, 65, 97, 94, 2, 93, 16, 67, 29, 62, 33, 24, 98, 5, 86, 35, 54, 0, 91, 52, 66, 85, 84, 56, 11, 1, 76, 25, 55, 21, 99, 72, 41, 23, 75, 28, 68, 69, 46, 8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [92, 82, 77, 64, 5, 33, 62, 56, 70, 0, 20, 28, 67, 14, 84, 53, 91, 29, 85, 2, 52, 83, 75, 35, 11, 21, 72, 98, 55, 1, 41, 76, 25, 66, 69, 9, 48, 54, 40, 23, 95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [47, 6, 19, 0, 62, 93, 59, 65, 54, 70, 34, 55, 23, 38, 72, 76, 53, 31, 78, 96, 77, 27, 92, 18, 82, 50, 98, 32, 1, 75, 83, 4, 51, 35, 80, 11, 74, 66, 36, 42, 33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],

    'tinyimgnet': [[0,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 15,  16,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28, 29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                      42,  43,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55, 56,  57,  59,  60,  61,  62,  63,  64,  65,  67,  68,  69,  70, 71,  73,  74,  76,  77,  78,  79,  80,  82,  83,  84,  85,  86,
                      87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127,
                      128, 129, 130, 131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 172, 173,
                      174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 198, 199],
                     [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  20,  21,  22,  24,  25,  26,  27,  28, 30,  31,  32,  33,  34,  36,  37,  39,  40,  41,  42,  43,  44,
                       45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58, 60,  62,  63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74, 75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,
                       89,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                       130, 131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                       172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197],
                     [1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  28, 29,  30,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
                      43,  44,  45,  47,  48,  49,  52,  53,  54,  55,  56,  57,  58, 59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72, 73,  74,  75,  76,  77,  79,  81,  82,  83,  84,  85,  86,  87,
                      88,  89,  90,  91,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                      131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                      174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199],
                      [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  14, 15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27, 28,  29,  30,  31,  32,  33,  34,  35,  37,  38,  39,  40,  41,
                      42,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55, 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68, 69,  70,  71,  72,  73,  74,  76,  77,  78,  79,  80,  81,  83,
                      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96, 97,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                      124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 159, 160, 161, 162, 163, 164, 165, 167,
                      168, 169, 171, 172, 173, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 192, 194, 195, 197, 198, 199],
                      [ 1,   2,   3,   4,   5,   7,   8,   9,  10,  11,  12,  13,  14, 15,  16,  17,  18,  19,  22,  24,  25,  27,  28,  29,  30,  32, 33,  34,  35,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
                        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59, 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72, 73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85,  86,
                       87,  88,  90,  91,  92,  93,  94,  96,  97,  98,  99, 100, 101, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 171, 172, 173,
                       174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]],
    
    "cub": [[20, 159, 173, 148, 1, 57, 113, 165, 52, 109, 14, 4, 180, 6, 182, 68, 
             33, 108, 46, 35, 75, 188, 187, 100, 47, 105, 41, 86, 16, 54, 139, 138], 
            [152, 195, 132, 83, 22, 192, 153, 175, 191, 155, 49, 194, 73, 66, 170, 151, 
             169, 96, 103, 37, 181, 127, 78, 21, 10, 164, 62, 2, 183, 85, 45, 60, 92, 185],
            [29, 110, 3, 8, 13, 58, 142, 25, 145, 63, 59, 65, 24, 140, 120, 32, 114, 107, 160,
             130, 118, 101, 115, 128, 117, 71, 156, 112, 36, 122, 104, 102, 90, 125]],

    "aircraft": [[6, 7, 8, 9, 12, 13, 31, 32, 25, 26, 18, 20, 15],
                 [78, 82, 51, 49, 50, 54, 55, 59, 60, 61, 68, 69, 70, 85, 86, 87, 88], 
                 [80, 81, 42, 84, 40, 90, 74, 75, 97, 98, 34, 35, 93, 94, 96, 72, 91, 83, 62, 89]],
    
    "stanfordcars": [[23, 42, 83, 94, 113, 126, 190],
                     [3, 8, 10, 17, 21, 27, 29, 45, 55, 96, 99, 103, 128, 179, 183],
                     [4, 5, 6, 12, 13, 14, 15, 18, 19, 24, 30, 31, 32, 33, 34, 35, 36,
                      37, 39, 40, 43, 47, 48, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62,
                      63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79,
                      80, 85, 86, 87, 88, 89, 90, 91, 92, 101, 106, 107, 108, 109, 110,
                      111, 114, 115, 116, 118, 119, 120, 121, 124, 130, 131, 132]],

    "cifar100_marco": [
                       #[4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
                       #[3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99],
                       [4,9,14,19,24,29,34,39,44,49,54,59,64,69,74,79,84,89,94,99]
    ]
}


def pickClass(classIdx):
    
    classNames = superClasses[classIdx]
    classList = []
    for n in classNames:
        classList.append(classMap[n])
        
    return classList


import copy
import random
from data_loader import iCIFAR10, iCIFAR100, TinyImagenet, customSVHN, mnist, CUB, Aircraft
from util import TwoCropTransform
from torchvision import transforms, datasets
from config import data_root
from augmentations.randaugment import augment_list, RandAugment
from augmentations.cut_out import *
from scipy.spatial.distance import mahalanobis
from util import  feature_stats
from PIL import Image
from util import accuracy_plain

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


num_inlier_classes_mapping = {"cifar10": 6, "cifar-10-100-10": 4, "cifar-10-100-50": 4, "cifar100_marco": 20,
                              "tinyimgnet": 20, "mnist": 6, "svhn": 6, "cub": 100, "aircraft": 20, "cifar100": 10}


data_function_mapping = {"cifar10": iCIFAR10, "cifar-10-100-10": iCIFAR10, "cifar-10-100-50": iCIFAR10, "cifar100_marco": iCIFAR100,
                         "cifar100": iCIFAR100, "tinyimgnet": TinyImagenet, "mnist": mnist, "svhn": customSVHN, "cub": CUB, "aircraft": Aircraft}

data_function_mapping_testing = {"cifar10": iCIFAR10, "cifar-10-100-10": iCIFAR100, "cifar-10-100-50": iCIFAR100, "cifar100_marco": iCIFAR100,
                                 "cifar100": iCIFAR100, "tinyimgnet": TinyImagenet, "mnist": mnist, "svhn": customSVHN, "cub": CUB, "aircraft": Aircraft}


mean_mapping = {"mnist":  (0.1307,),
                "svhn": (0.4376821, 0.4437697, 0.47280442),
                "cifar10": (0.4914, 0.4822, 0.4465),
                "cifar100": (0.4914, 0.4822, 0.4465),
                "cifar100_marco": (0.4914, 0.4822, 0.4465),
                "cifar-10-100-10": (0.4914, 0.4822, 0.4465),
                "cifar-10-100-50": (0.4914, 0.4822, 0.4465),
                "tinyimgnet": (0.485, 0.456, 0.406),
                "aircraft": (0.485, 0.456, 0.406), 
                "cub": (0.485, 0.456, 0.406)}                 # 0.408, 0.459, 0.502, 123., 117., 104.

std_mapping = {"mnist": (0.3081,),
               "svhn": (0.19803012, 0.20101562, 0.19703614),
               "cifar10": (0.2023, 0.1994, 0.2010),
               "cifar100": (0.2023, 0.1994, 0.2010),
               "cifar100_marco": (0.2023, 0.1994, 0.2010),
               "cifar-10-100-10": (0.2023, 0.1994, 0.2010),
               "cifar-10-100-50": (0.2023, 0.1994, 0.2010),
               "tinyimgnet": (0.229, 0.224, 0.225),
               "aircraft": (0.229, 0.224, 0.225),
               "cub": (0.229, 0.224, 0.225)}               # 1., 1., 1.


image_size_mapping = {"mnist": 32,
                      "svhn": 32,
                      "cifar10": 32,
                      "cifar100": 32,
                      "cifar100_marco": 32,
                      "cifar-10-100-10": 32,
                      "cifar-10-100-50": 32,
                      "tinyimgnet": 64, 
                      "aircraft": 224,
                      "cub": 224}


def label_to_dict(labels, outliers=False):
    label_dict = dict()
    for i, l in enumerate(labels):
        if outliers is False:
            label_dict[str(l)] = i
        else:
            label_dict[str(l)] = 1000

    return label_dict


def get_train_datasets(opt, class_idx=None, last_features_list=None, last_feature_labels_list=None, last_model=None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.action == "training_supcon" or opt.action == "trainging_linear":
        if opt.datasets == "mnist":
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation((-5, 5)),])
                                                                            
        else:
            train_transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),                                     
                                                  transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                                  #cutout(mask_size=4, p=0.5, cutout_inside=False),
                                                  transforms.ToTensor(),
                                                  normalize,])   # normalize,
            #train_transform.transforms.insert(0, RandAugment(args=opt))       # !!!!
    elif opt.action == "training_ce":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        if opt.datasets == "mnist":
            train_transform = transforms.Compose([transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if last_features_list is not None and last_feature_labels_list is not None:
          subsample_transform = transforms.Compose([transforms.RandomResizedCrop(size=size, ),
                                                    transforms.ToTensor(), normalize,])
    else:
          subsample_transform = None

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if opt.action == "training_supcon":
        train_transform = TwoCropTransform(train_transform)
    
    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    if opt.datasets == "svhn":
        train = "train"
    else:
        train = True

    train_dataset = data_fun(root=data_root, train=train,
                             classes=classes, download=False, 
                             transform=train_transform, label_dict=label_dict,
                             )
    print("dataset size", len(train_dataset))
    return train_dataset


def get_test_datasets(opt, class_idx = None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.datasets == "mnist":
        test_transform = transforms.Compose([transforms.ToTensor(), ]) 
    else:
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    if opt.datasets == "svhn":
        train = "test"
    else:
        train = False
    test_dataset = data_fun(root=data_root, train=train,
                            classes=classes, download=True, 
                            transform=test_transform, label_dict=label_dict)
    print("dataset size", len(test_dataset))
    return test_dataset


def get_outlier_datasets(opt, class_idx=None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.action == "outlier_visualization":
        test_transform = transforms.Compose([transforms.ToTensor(),])
    else:
        if opt.datasets == "mnist":
            test_transform = transforms.Compose([transforms.ToTensor(),])
        else:
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping_testing[opt.datasets]
    label_dict = label_to_dict(osr_splits_outliers[opt.datasets][opt.trail], outliers=True)
    if class_idx is not None:
        classes = [osr_splits_outliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_outliers[opt.datasets][opt.trail]
    print(classes)
    if opt.datasets == "svhn":
        train = "test"
    else:
        train = False
    outlier_dataset = data_fun(root=data_root, train=train,
                               classes=classes, download=True, 
                               transform=test_transform, label_dict=label_dict)
    print("dataset size", len(outlier_dataset))
    return outlier_dataset


def sortFeatures(mixedFeatures, labels, num_classes):
        
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])

    print("mixedFeatures", np.array(mixedFeatures).shape)
    
    for i, l in enumerate(labels):
        l = l.item()                         
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures


def mahalanobis_group(features, mu, var):

    dis = []
    for feature in features:
        d = mahalanobis(feature, mu, np.linalg.inv(var))
        #print("feature, mu", np.mean(feature), np.mean(mu), np.mean(feature-mu), np.matmul((feature-mu), (feature-mu).T), d)
        dis.append(d)
    
    return dis


def mahalanobis_stats(sorted_last_features, stats, ratio_in, ratio_out):
    
    cut_thresholds = []
    for c, (last_features_c, stats_c) in enumerate(zip(sorted_last_features, stats)):
        mu, var = stats_c
        m_distances = mahalanobis_group(last_features_c, mu, var)        
        m_distances = np.sort(m_distances)
        print("m_distances", np.mean(m_distances))
        cut_index1 = int((1-ratio_out)*len(m_distances))   # outside
        cut_threshold1 = m_distances[cut_index1]
        cut_index2 = int(ratio_in*len(m_distances))         # centers
        cut_threshold2 = m_distances[cut_index2]
        cut_thresholds.append((cut_threshold1, cut_threshold2))

    return cut_thresholds


def normalFeatureReading(data_loader, model, opt):
    
    outputs_backbone = []
    outputs = []
    outputs_linear = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if i > 5000:
            break

        if opt.method == "SupCon":
            output, output_encoder = model(img), model.encoder(img)
        else:
            output = model.encoder(img)

        outputs.append(output.detach().numpy())
        outputs_backbone.append(output_encoder.detach().numpy())

        labels.append(label.numpy())
    
    return output, output_encoder, labels


def featureMerge(featureList):
    
    featureMaps = []
    featureMaps_backbone = []
    labels = []

    for features in featureList:
        
        features, feature_backbone, labels_part = features
  
        featureMaps_backbone = featureMaps_backbone + feature_backbone
        featureMaps = featureMaps + features
        labels = labels + labels_part
        
    featureMaps_backbone = np.array(featureMaps_backbone, dtype=object)
    featureMaps = np.array(featureMaps, dtype=object)
    featureMaps_backbone = np.squeeze(featureMaps_backbone)
    featureMaps = np.squeeze(featureMaps)
    labels = np.squeeze(np.array(labels))

    return featureMaps, featureMaps_backbone, labels
    

def linear_classifier(mixedFeatures_train, labels_train, mixedFeatures_test, labels_test):

    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(mixedFeatures_train, labels_train)

    test_pred = clf.predict(mixedFeatures_test)
    print("Linear Classifier Accuracy: ", accuracy_plain(test_pred, labels_test))


def process_double_label(double_label):

    label_len = len(double_label)
    mask = np.ones([label_len, label_len])

    for i, l1 in enumerate(double_label):
        for j, l2 in enumerate(double_label):
            if len(list(set(l1).intersection(l2))) > 0:
                mask[i, j] = 0

    return mask


def similarity_to_probability(similarity_matrix):

    """
    Here set small similarity with larger probability
    """

    #print("similarity_matrix", similarity_matrix)
    similarity_logit = np.exp(-similarity_matrix)
    np.fill_diagonal(similarity_logit, 0)
    probability_matrix = similarity_logit / np.sum(similarity_logit, axis=1, keepdims=True)

    return probability_matrix


    
def binary_random(p):

     """
     the probability of return 1 is p
     """
     return np.random.binomial(1, p)

    
def selection_with_similarity(similarity_matrix, p):

    idxs = []
    for i, ss in enumerate(similarity_matrix):
        
        if binary_random(p):
            idx = np.argmin(ss)
        else:
            idx = i
        
        idxs.append(idx)
    
    return idxs


def selection_random(num_sample):

    idxs = []
    for i in range(num_sample):
        idxs.append(random.randrange(0, num_sample))

    return idxs

