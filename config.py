# Constant Variables

# -----------------------
# DATASET ROOT DIRS
# -----------------------
data_root = "../datasets"                    


# augmentation
augmentation_mapping = {"svhn":    [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                   ["Rotate"], 
                                   ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "cifar10": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                    ["Rotate"], 
                                    ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "cifar100": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                     ["Rotate"], 
                                     ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],

                        "tinyimagenet": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                         ["Rotate"], 
                                         ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],
                                         
                        "cub": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                ["Rotate"], 
                                ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]],
                        
                        "aircraft": [["ShearX", "ShearY", "TranslateX", "TranslateY", "TranslateXabs"], 
                                     ["Rotate"], 
                                     ["AutoContrast", "Posterize", "Contrast", "Brightness", "Sharpness"]]}


temperature1_scheduling_mapping = {"mnist": [1.0, 0.5, 0.05],  
                                  "svhn": [0.1, 0.05, 0.01, 0.005],
                                  "cifar10": [0.1, 0.01, 0.005], 
                                  "cifar-10-100-10": [0.1, 0.05, 0.01],
                                  "cifar-10-100-50": [0.1, 0.05, 0.01],
                                  "tinyimagenet": [0.1, 0.05, 0.01]}

temperature2_scheduling_mapping = {"mnist": [1.0, 0.5, 0.05],  
                                  "svhn": [0.1, 0.05, 0.01, 0.005],
                                  "cifar10": [1., 1., 1.], 
                                  "cifar-10-100-10": [0.1, 0.05, 0.01],
                                  "cifar-10-100-50": [0.1, 0.05, 0.01],
                                  "tinyimagenet": [0.1, 0.05, 0.01]}



temperature_scheduling_epoch_mapping = {"mnist": [0, 50, 150],  
                                        "svhn": [0, 100, 200, 300],
                                        "cifar10": [0, 10, 20],    
                                        "cifar-10-100-10": [0, 100, 200],
                                        "cifar-10-100-50": [0, 100, 200],
                                        "tinyimagenet": [0, 100, 200]}

sampling_scheduling_epoch_mapping = {"mnist": [200],  
                                     "svhn": [200],
                                     "cifar10": [200],
                                     "cifar-10-100-10": [200],
                                     "cifar-10-100-50": [200],
                                     "tinyimagenet": [200]}

