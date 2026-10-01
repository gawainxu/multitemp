import os
import sys
sys.path.append("../")
import numpy as np
import math
import pdb

import util

def get_classes(items):
    shapes, textures = [], []
    for item in items:
        shapes.append(item["shape"])
        textures.append(item["texture"])
    return set(shapes), set(textures)

def all_classes_present(candidates, shapes, textures):
    cand_shapes, cand_textures = get_classes(candidates)
    missing_shapes = shapes - cand_shapes
    missing_textures = textures - cand_textures
    if len(missing_shapes) > 0 or len(missing_textures) > 0:
        print("missing: {}, {}".format(missing_shapes, missing_textures))
        return False
    return True

def sample_perc_subsets(versions_to_splits, percs):
    # ensure that all shape, texture classes that appear in train set 
    # appear at least once per subset
    out = dict(zip(versions_to_splits.keys(), 
                  [dict() for _ in range(len(versions_to_splits))]))
    for version_name, splits in versions_to_splits.items():
        out[version_name]["val"], out[version_name]["train"] = splits["val"], dict()
        shapes, textures = get_classes(splits["train"])
        train_size = len(splits["train"])
        avail_inds = range(train_size)
        for p in percs:
            if p == 1.0:
                out[version_name]["train"][p] = splits["train"]
            else:
                num_items = int(math.floor(p * train_size))
                attempts = 0
                while True:
                    sampled_item_inds = np.random.choice(avail_inds, num_items, 
                                                         replace=False)
                    candidates = [splits["train"][i] for i in sampled_item_inds]
                    attempts += 1
                    if all_classes_present(candidates, shapes, textures):
                        out[version_name]["train"][p] = candidates
                        print("created in {} attempts".format(attempts))
                        break
                    if attempts > 500:
                        print("not possible")
                        pdb.set_trace()
    return out

def main(versions_to_splits_path, percs, save_path):
    versions_to_splits = util.load_pkl(versions_to_splits_path)
    versions_to_splits_to_percents = sample_perc_subsets(versions_to_splits, 
                                                         percs)
    util.save_pkl(save_path, versions_to_splits_to_percents)
    print("\nSaved\n")

if __name__=="__main__":
    # splits_dir = "dataset_specs/navon_twice_rotated/"
    # percs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for task in ["shape", "texture"]:
    #     pth = os.path.join("", splits_dir, "{}_splits".format(task))
    #     main(os.path.join(pth, "splits.pkl"), percs, os.path.join(pth, "percents"))

    # cst_correlated_rt = "/data2/lampinen/shared_structure/dataset_specs/"
    # base = "cst_correlated-shape-texture_cond_prob-"
    # # base = "cst_correlated-shape-color_cond_prob-"
    # for prob in ["{:.2}".format(p).replace(".", "o") for p in np.arange(0.1, 1.1, 0.1)]:
    #     splits_dir = os.path.join(cst_correlated_rt, base + prob)
    #     for task in ["shape", "texture", "color"]:
    #         pth = os.path.join("", splits_dir, "{}_splits".format(task))
    #         main(os.path.join(pth, "splits.pkl"), [1.0], os.path.join(pth, "percents"))

    # splits_dir = "/data2/lampinen/shared_structure/dataset_specs/cst"
    splits_dir = "/data2/lampinen/shared_structure/dataset_specs/cst_larger" #cst_fullset"
    for task in ["shape", "texture", "color"]:
        pth = os.path.join("", splits_dir, "{}_splits".format(task))
        main(os.path.join(pth, "splits.pkl"), [1.0], os.path.join(pth, "percents"))


