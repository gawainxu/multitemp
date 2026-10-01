import os
import sys
sys.path.append("../")
import random
import re
import pdb

from collections import Counter

import util


"""Utils."""
def print_dict(d, log_path):
    for k, v in d.items():
        util.print_log("{}: {}".format(k, v), log_path)


def print_lens(d, log_path):
    for k, v in d.items():
        util.print_log("{}: {}".format(k, len(v)), log_path)


def count_pairings(splits):
    """Count number of time  a given shape (any exemplar) and texture are
    paired in each set.
    """
    def set_default():
        return {"full": 0, "val": 0, "train": 0}
    
    pairing_to_split_to_count = dict()
    for item, which_split in zip(splits["val"] + splits["train"], 
                                ["val" for _ in range(len(splits["val"]))] + 
                                ["train" for _ in range(len(splits["train"]))]):
        k = "{}-{}".format(item["shape"], item["texture"])
        pairing_to_split_to_count.setdefault(k, set_default())
        pairing_to_split_to_count[k]["full"] += 1
        pairing_to_split_to_count[k][which_split] += 1

    out_str = "\n\nShape class-texture class pairings:\n"
    in_val, in_train = 0, 0
    for pairing, v in pairing_to_split_to_count.items():
        out_str += "\n{}: {}/{} ({:.3}) in val".format(
                        pairing, v["val"], v["full"], v["val"]/v["full"])
        if v["val"] > 0:
            in_val += 1
        if v["train"] > 0:
            in_train +=1
    out_str += "\n\n{}/{} ({:.3}) pairings appear in validation set".format(
                in_val, len(pairing_to_split_to_count),
                in_val/len(pairing_to_split_to_count))
    out_str += "\n{}/{} ({:.3}) pairings appear in train set".format(
                in_train, len(pairing_to_split_to_count), 
                in_train/len(pairing_to_split_to_count))
    return out_str


def count_holdouts(splits, holdout_shape_classes, holdout_texture_classes):
    val_shapes, val_textures = [], []
    train_shapes, train_textures = [], []
    for item, which_split in zip(splits["val"] + splits["train"], 
                                ["val" for _ in range(len(splits["val"]))] + 
                                ["train" for _ in range(len(splits["train"]))]):
        if which_split == "val":
            val_shapes.append(item["shape"])
            val_textures.append(item["texture"])
        elif which_split == "train":
            train_shapes.append(item["shape"])
            train_textures.append(item["texture"])
    val_shapes, val_textures = set(val_shapes), set(val_textures)
    train_shapes, train_textures = set(train_shapes), set(train_textures)

    heldout_shapes = val_shapes - train_shapes
    heldout_textures = val_textures - train_textures

    heldout_shapes_beyond_targets = heldout_shapes - set(holdout_shape_classes)
    heldout_textures_beyond_targets = heldout_textures - set(holdout_texture_classes)

    out_str = "\n\nWhat ends up held out:"
    out_str += "\nHeldout shapes: {}".format(heldout_shapes)
    out_str += "\nHeldout shape classes not in orig target holdouts: {}".format(
                    heldout_shapes_beyond_targets)
    out_str += "\nHeldout textures: {}".format(heldout_textures)
    out_str += "\nHeldout texture classes not in orig target holdouts: {}".format(
                    heldout_shapes_beyond_targets)
    return out_str


def log_stats(splits, task, holdout_shape_classes, holdout_texture_classes,
              log_path):
    out_str = ""  
    out_str += "\nTask: {}".format(task) 
    out_str += "\nHeld out from train set:"
    out_str += "\n{} shape classes: {}".format(len(holdout_shape_classes), 
                                               holdout_shape_classes)
    out_str += "\n{} texture classes: {}".format(len(holdout_texture_classes),
                                                 holdout_texture_classes)
    out_str += count_pairings(splits)
    out_str += count_holdouts(splits, holdout_shape_classes, 
                              holdout_texture_classes)
    tot_size = len(splits["val"]) + len(splits["train"])
    out_str += "\n\nDataset size:"
    out_str += "\nValidation size = {}/{} ({:.3})".format(
                 len(splits["val"]), tot_size, len(splits["val"])/tot_size)
    out_str += "\n\nTrain size = {}/{}".format(len(splits["train"]), tot_size)

    util.print_log(out_str, log_path)


"""Utils: navon dataset."""
def navon_fname_to_item_dict(fname):
    shape_exemplar, texture_exemplar = fname.split("-")
    texture_exemplar = texture_exemplar.split(".")[0]

    item = dict()
    item["shape"] = shape_exemplar.split("_")[0]
    item["texture"] = texture_exemplar
    item["shape_exemplar"] = shape_exemplar
    item["texture_exemplar"] = None
    item["fname"] = fname
    return item


def get_navon_items(dataset_dir):
    items = []
    shape_dirs = util.list_dir(dataset_dir)
    for shape_dir in shape_dirs:
        items.extend(util.list_dir(os.path.join(dataset_dir, shape_dir)))
    items = [*map(navon_fname_to_item_dict, items)]

    out = []
    for item in items:
        if item["shape"] == item["texture"]:
            print("Excluding {}".format(item))
        else:
            out.append(item)
    return out


def classes_avail(items, class_type):
    assert class_type in ["shape", "texture"]
    classes = []
    for item in items:
        classes.append(item[class_type])
    return [*set(classes)]


def new_version(items, shape_classes, texture_classes, num_holdout_classes,
                shape_log_path, texture_log_path):
    # Sample hold-out classes.
    holdout_shape_classes = random.sample(shape_classes, 
                                          k=num_holdout_classes)
    holdout_texture_classes = random.sample(texture_classes, 
                                            k=num_holdout_classes)

    # print("held-out shapes: {}".format(holdout_shape_classes))
    # print("held-out textures: {}".format(holdout_texture_classes))

    # Assemble train set (shared by both tasks) and task-specific val sets
    train_set, shape_task_val_set, texture_task_val_set = [], [], []
    discards = []
    for item in items:
        if (item["shape"] in holdout_shape_classes and 
            item["texture"] in holdout_texture_classes):
            # print("holding out shape={}, texture={}".format(item["shape"], 
            #                                                 item["texture"]))
            discards.append(item)
        elif item["shape"] in holdout_shape_classes:
            # Goes to texture task val set (where we want to test generalization 
            # across shapes).
            texture_task_val_set.append(item)

        elif item["texture"] in holdout_texture_classes:
            # Goes to shape task val set (where we want to test generalization 
            # across textures).
            shape_task_val_set.append(item)
        else:
            train_set.append(item)
    print("{} discards".format(len(discards)))

    shape_task_dataset = {"train": train_set, "val": shape_task_val_set,
                          "holdout_classes": holdout_texture_classes}
    texture_task_dataset = {"train": train_set, "val": texture_task_val_set,
                            "holdout_classes": holdout_shape_classes}

    # Record.
    log_stats(shape_task_dataset, "shape", holdout_shape_classes, 
              holdout_texture_classes, shape_log_path)
    log_stats(texture_task_dataset, "texture", holdout_shape_classes,
              holdout_texture_classes, texture_log_path)

    return shape_task_dataset, texture_task_dataset


def multiversion_splits_navon(dataset_dir, save_dir, num_versions=5,
                              num_holdout_classes=3):
    """For each version, holds out num_holdout_classes shape classes and
    num_holdout classes texture classes. 

    The remaining items comprise the train set for shape and texture tasks. 
    
    For the shape task, the held-out texture classes are used as the 
    validation set; for the texture task, the held-out shape classes are. 

    We ensure that the held-out shape and texture classes are not crossed 
    with each other in any item (i.e. all instances of a held-out shape are 
    paired with textures seen during training, and vice versa) by discarding them.

    Args: 
        dataset_dir: str, src dataset directory (containing images).
    """

    # File handling.
    shape_save_dir = os.path.join(save_dir, "shape_splits")
    texture_save_dir = os.path.join(save_dir, "texture_splits")
    shape_log_dir = os.path.join(shape_save_dir, "logs")
    texture_log_dir = os.path.join(texture_save_dir, "logs")
    util.make_dirs([save_dir, shape_save_dir, texture_save_dir,
                    shape_log_dir, texture_log_dir])

    # Read in dataset.
    items = get_navon_items(dataset_dir)
    shape_classes = classes_avail(items, "shape")
    texture_classes = classes_avail(items, "texture")

    cv_splits_shape, cv_splits_texture = dict(), dict()
    for version in range(num_versions):
        version_name = "version_{}".format(version)
        print("\nMaking + splitting {}".format(version_name))
        shape_log_path = os.path.join(shape_log_dir, "{}_split.txt".format(version_name))
        texture_log_path = os.path.join(texture_log_dir, "{}_split.txt".format(version_name))

        (shape_task_dataset, 
          texture_task_dataset) = new_version(items, shape_classes, texture_classes, 
                                              num_holdout_classes, shape_log_path,
                                              texture_log_path)
        cv_splits_shape[version_name] = shape_task_dataset
        cv_splits_texture[version_name] = texture_task_dataset

    util.save_pkl(os.path.join(shape_save_dir, "splits"), cv_splits_shape)
    util.save_pkl(os.path.join(texture_save_dir, "splits"), cv_splits_texture)
    print("\n\nFinished splitting navon\n")


def main_navon():
    save_dir = "dataset_specs/navon_twice_rotated/"
    navon_stims_dir = "navon_twice_rotated"

    # File handling.
    util.make_dirs([save_dir])

    multiversion_splits_navon(navon_stims_dir, save_dir, num_holdout_classes=3)


"""Utils for color-shape-texture (CST) dataset."""
def cst_fname_to_item_dict(fname):
    shape, texture, color, exemplar_num = fname.split("_")
    exemplar_num = exemplar_num.split(".")[0]

    item = dict()
    item["shape"] = shape
    item["texture"] = texture
    item["color"] = color
    item["exemplar"] = exemplar_num
    item["fname"] = fname
    return item


def get_cst_items(dataset_dir):
    items = util.list_dir(dataset_dir)
    items = [*map(cst_fname_to_item_dict, items)]
    return items 


def classes_avail(items, class_type):
    assert class_type in ["shape", "texture", "color"]
    classes = []
    for item in items:
        classes.append(item[class_type])
    return [*set(classes)]


def log_stats_cst(splits, task, holdout_shape_classes, holdout_texture_classes,
                  holdout_color_classes, log_path):
    out_str = ""  
    out_str += "\nTask: {}".format(task) 
    out_str += "\nHeld out from train set:"
    out_str += "\n{} shape classes: {}".format(len(holdout_shape_classes), 
                                               holdout_shape_classes)
    out_str += "\n{} color classes: {}".format(len(holdout_color_classes), 
                                               holdout_color_classes)
    out_str += "\n{} texture classes: {}".format(len(holdout_texture_classes),
                                                 holdout_texture_classes)
#    out_str += count_pairings(splits)
#    out_str += count_holdouts(splits, holdout_shape_classes, 
#                              holdout_texture_classes)
    tot_size = len(splits["val"]) + len(splits["train"])
    out_str += "\n\nDataset size:"
    out_str += "\nValidation size = {}/{} ({:.3})".format(
                 len(splits["val"]), tot_size, len(splits["val"])/tot_size)
    out_str += "\n\nTrain size = {}/{}".format(len(splits["train"]), tot_size)

    util.print_log(out_str, log_path)


def new_version_cst(items, shape_classes, texture_classes, color_classes,
                    num_holdout_classes, shape_log_path, texture_log_path,
                    color_log_path, preserve_correlations,
                    restrict_train_set_to, holdouts_on_target_attribute):
    # Sample hold-out classes.
    if preserve_correlations:
        holdout_indices = random.sample(range(len(shape_classes)),
                                        k=num_holdout_classes) 
        holdout_shape_classes = [shape_classes[i] for i in holdout_indices]
        holdout_texture_classes = [texture_classes[i] for i in holdout_indices]
        holdout_color_classes = [color_classes[i] for i in holdout_indices]
    else:
        holdout_shape_classes = random.sample(shape_classes, 
                                              k=num_holdout_classes)
        holdout_texture_classes = random.sample(texture_classes, 
                                                k=num_holdout_classes)
        holdout_color_classes = random.sample(color_classes, 
                                              k=num_holdout_classes)

    print("held-out colors: {}".format(holdout_color_classes))
    print("held-out shapes: {}".format(holdout_shape_classes))
    print("held-out textures: {}".format(holdout_texture_classes))

    # Assemble train set (shared by all tasks) and task-specific val sets.
    (train_set, shape_task_train_set, 
     texture_task_train_set, color_task_train_set,
     shape_task_val_set, texture_task_val_set, 
     color_task_val_set) = [], [], [], [], [], [], []
    discards = []
    for item in items:
        # Note: this code allows for hold-outs to differ along both other
        # dimensions -- it would also be possible to allow only one 
        # dimension to differ, which would make for less challenging eval.
        if holdouts_on_target_attribute:
            if (item["shape"] in holdout_shape_classes and 
                item["texture"] in holdout_texture_classes and
                item["color"] in holdout_color_classes):
                discards.append(item)
            elif (item["shape"] not in holdout_shape_classes and 
                  item["texture"] not in holdout_texture_classes and
                  item["color"] not in holdout_color_classes):
                train_set.append(item)
            else:
                    if item["shape"] not in holdout_shape_classes:
                        shape_task_val_set.append(item)
                    if item["color"] not in holdout_color_classes:
                        color_task_val_set.append(item)
                    if item["texture"] not in holdout_texture_classes:
                        texture_task_val_set.append(item)

        else: # Not holdouts_on_target_attribute:  
            if (item["shape"] not in holdout_shape_classes and 
                  item["texture"] not in holdout_texture_classes and
                  item["color"] not in holdout_color_classes):
                train_set.append(item)
            else:
                # Add holdout classes for attribute to specific train set.
                if (item["shape"] not in holdout_shape_classes and
                    item["color"] not in holdout_color_classes):
                    texture_task_train_set.append(item)
                elif (item["texture"] not in holdout_texture_classes and
                      item["color"] not in holdout_color_classes):
                    shape_task_train_set.append(item)
                elif (item["texture"] not in holdout_texture_classes and
                      item["shape"] not in holdout_shape_classes):
                    color_task_train_set.append(item)

                if item["shape"] in holdout_shape_classes:
                    color_task_val_set.append(item)
                    texture_task_val_set.append(item)
                if item["color"] in holdout_color_classes:
                    shape_task_val_set.append(item)
                    texture_task_val_set.append(item)
                if item["texture"] in holdout_texture_classes:
                    color_task_val_set.append(item)
                    shape_task_val_set.append(item)

    
    total_shape_train_set = train_set + shape_task_train_set
    total_color_train_set = train_set + color_task_train_set
    total_texture_train_set = train_set + texture_task_train_set
    if restrict_train_set_to is not None:
        if holdouts_on_target_attribute:
            random.shuffle(train_set)
            train_set = train_set[:restrict_train_set_to]
            print("{} train".format(len(train_set)))
            # Ensure all have the same train set.
            total_shape_train_set = train_set
            total_color_train_set = train_set
            total_texture_train_set = train_set
        else:
            random.shuffle(total_shape_train_set)
            total_shape_train_set = total_shape_train_set[:restrict_train_set_to]
            print("{} shape train".format(len(total_shape_train_set)))
            random.shuffle(total_color_train_set)
            total_color_train_set = total_color_train_set[:restrict_train_set_to]
            print("{} color train".format(len(total_color_train_set)))
            random.shuffle(total_texture_train_set)
            total_texture_train_set = total_texture_train_set[:restrict_train_set_to]
            print("{} texture train".format(len(total_texture_train_set)))
    print("{} discards".format(len(discards)))
    print("{} texture val".format(len(texture_task_val_set)))
    print("{} shape val".format(len(shape_task_val_set)))
    print("{} color val".format(len(color_task_val_set)))

    shape_task_dataset = {"train": total_shape_train_set,
                          "val": shape_task_val_set,
                          "holdout_classes": holdout_texture_classes + holdout_color_classes}
    color_task_dataset = {"train": train_set + color_task_train_set,
                          "val": color_task_val_set,
                          "holdout_classes": holdout_texture_classes + holdout_shape_classes}
    texture_task_dataset = {"train": train_set + texture_task_train_set,
                            "val": texture_task_val_set,
                            "holdout_classes": holdout_shape_classes + holdout_color_classes}

    # Record.
    log_stats_cst(shape_task_dataset, "shape", holdout_shape_classes, 
                  holdout_texture_classes, holdout_color_classes,
                  shape_log_path)
    log_stats_cst(color_task_dataset, "color", holdout_shape_classes, 
                  holdout_texture_classes, holdout_color_classes,
                  color_log_path)
    log_stats_cst(texture_task_dataset, "texture", holdout_shape_classes,
                  holdout_texture_classes, holdout_color_classes,
                  texture_log_path)

    return shape_task_dataset, color_task_dataset, texture_task_dataset


def multiversion_splits_cst(dataset_dir, save_dir, num_versions=5,
                            num_holdout_classes=3, correlated_features=None,
                            corr_feat_cond_prob=None,
                            restrict_train_sets_to=None,
                            holdouts_on_target_attribute=True):
    """For each version, holds out num_holdout_classes shape classes and
    num_holdout classes texture classes. 

    The remaining items comprise the train set for shape and texture tasks. 
    
    For the shape task, the held-out texture classes are used as the 
    validation set; for the texture task, the held-out shape classes are. 

    We ensure that the held-out shape and texture classes are not crossed 
    with each other in any item (i.e. all instances of a held-out shape are 
    paired with textures seen during training, and vice versa) by discarding them.

    Args: 
        dataset_dir: src dataset directory (containing images)
        correlated_features: list containing subset of ["shape", "color",
            "texture"], or None -- which features should correlate
        corr_feat_cond_prob: If features are correlated, what the
            conditional probability of one feature given another should
            be. E.g. at 0.9, 90\% will match. Setting this to 0.1 produces
            uncorrelated features (as does setting correlated_features=None).
            If three featuers are correlated, this is the probability that all
            three match. Conditioned on all three not matching, it is then the
            probability that two will match. That is, with three correlated,
            p(none match) = (1 - corr_feat_cond_prob)^2
        restrict_train_sets_to: If not None, how many train items to use (to
            match dataset sizes across different correlations, e.g.)
        holdouts_on_target_attribute: Whether to hold-out classes from target
            attribute or not. Doing so makes datasets more matched across
            different target attributes, not doing so gives more classes in the
            target attribute, so can give better inferences.
    """

    # File handling.
    shape_save_dir = os.path.join(save_dir, "shape_splits")
    color_save_dir = os.path.join(save_dir, "color_splits")
    texture_save_dir = os.path.join(save_dir, "texture_splits")
    shape_log_dir = os.path.join(shape_save_dir, "logs")
    color_log_dir = os.path.join(color_save_dir, "logs")
    texture_log_dir = os.path.join(texture_save_dir, "logs")
    util.make_dirs([save_dir, shape_save_dir, color_save_dir, texture_save_dir,
                    shape_log_dir, color_log_dir, texture_log_dir])

    # Read in dataset.
    items = get_cst_items(dataset_dir)
    shape_classes = classes_avail(items, "shape")
    color_classes = classes_avail(items, "color")
    texture_classes = classes_avail(items, "texture")


    if correlated_features is not None:
        if corr_feat_cond_prob is None:
            raise ValueError("You must specify a feature conditional"
                             "probability with correlated features.")
        # Shuffle the features, so matches will be shared across different
        # splits, but not across different runs of this code.
        random.shuffle(shape_classes)
        random.shuffle(color_classes)
        random.shuffle(texture_classes)

        # Create the assignment of which features will match.
        correlation_s_feature_matches = {s: (c, t) for (s, c, t) in zip(
            shape_classes, color_classes, texture_classes)}
        correlation_feature_ct_matches = {c: t for (c, t) in zip(
            color_classes, texture_classes)}

        three_feat_pair_match_prob = corr_feat_cond_prob * (1 - corr_feat_cond_prob)
        three_feat_none_match_prob = (1 - corr_feat_cond_prob)**2

        # Check feature correlations (pre).
        def feature_matches(item):
            match_ct = correlation_feature_ct_matches[item["color"]] 
            match_ct = match_ct == item["texture"]
            match_sc, match_st = correlation_s_feature_matches[item["shape"]] 
            match_sc = match_sc == item["color"]
            match_st = match_st == item["texture"]
            if match_sc and match_st:
                return "all"
            elif match_sc:
                return "shape_color"
            elif match_st:
                return "shape_texture"
            elif match_ct:
                return "color_texture"
            return "none"
        print("Feature co-occurence counts (before correlating):")
        print(Counter([feature_matches(item) for item in items]))       
       
        # Correlate features by first creating lookup dict by type:
        stimulus_type_dict = {x: [] for x in ["all", "shape_color", "shape_texture", "color_texture", "none"]}
        for item in items:
            stimulus_type_dict[feature_matches(item)].append(item)

        # Now do an appropriate filtering for what is correlated
        # note that this code may not work properly if there are different
        # numbers of clases of each feature type, or if items are unbalanced.
        if len(correlated_features) == 3:  # all correlated
            items = stimulus_type_dict["all"]
            num_pos = len(items)
            num_neg = num_pos * (1 - corr_feat_cond_prob) / (corr_feat_cond_prob)
            num_part_match_per = int(num_neg * corr_feat_cond_prob / 3.) 
            num_no_match = int(num_neg * (1 - corr_feat_cond_prob))
            for pair_type in ["shape_color", "shape_texture", "color_texture"]:
                random.shuffle(stimulus_type_dict[pair_type])
                items += stimulus_type_dict[pair_type][:num_part_match_per]
            random.shuffle(stimulus_type_dict["none"])
            items += stimulus_type_dict["none"][:num_no_match]
        else:
            match_pair_type = [k for k in stimulus_type_dict.keys() if all([f in k for f in correlated_features])]
            items = stimulus_type_dict["all"] + stimulus_type_dict[match_pair_type[0]]
            num_pos = len(items)
            num_neg = int(num_pos * (1 - corr_feat_cond_prob) / (corr_feat_cond_prob))
            neg_items = stimulus_type_dict["none"]
            for pair_type in ["shape_color", "shape_texture", "color_texture"]:
                if all([f in pair_type for f in correlated_features]):
                    continue
                neg_items += stimulus_type_dict[pair_type]
            random.shuffle(neg_items)
            items += neg_items[:num_neg]

        # Check feature correlations (post).
        print("Feature co-occurence counts (after correlating):")
        print(Counter([feature_matches(item) for item in items]))       

    cv_splits_shape, cv_splits_color, cv_splits_texture = dict(), dict(), dict()
    for version in range(num_versions):
        version_name = "version_{}".format(version)
        print("\nMaking + splitting {}".format(version_name))
        shape_log_path = os.path.join(shape_log_dir, "{}_split.txt".format(version_name))
        color_log_path = os.path.join(color_log_dir, "{}_split.txt".format(version_name))
        texture_log_path = os.path.join(texture_log_dir, "{}_split.txt".format(version_name))

        (shape_task_dataset, 
         color_task_dataset,
         texture_task_dataset) = new_version_cst(
            items, shape_classes, texture_classes, color_classes, 
            num_holdout_classes, shape_log_path, texture_log_path,
            color_log_path,
            preserve_correlations=correlated_features is not None,
            restrict_train_set_to=restrict_train_sets_to,
            holdouts_on_target_attribute=holdouts_on_target_attribute)
        cv_splits_shape[version_name] = shape_task_dataset
        cv_splits_color[version_name] = color_task_dataset
        cv_splits_texture[version_name] = texture_task_dataset

    util.save_pkl(os.path.join(shape_save_dir, "splits"), cv_splits_shape)
    util.save_pkl(os.path.join(color_save_dir, "splits"), cv_splits_color)
    util.save_pkl(os.path.join(texture_save_dir, "splits"), cv_splits_texture)
    print("\n\nFinished splitting cst\n")

    return len(shape_task_dataset["train"])


def main_cst():
    save_dir = "/data2/lampinen/shared_structure/dataset_specs/cst_larger"
    #save_dir = "/data2/lampinen/shared_structure/dataset_specs/cst"
    cst_stims_dir = "/data2/lampinen/color_texture_shape_stimuli"

    # file handling
    util.make_dirs([save_dir])

    multiversion_splits_cst(cst_stims_dir, save_dir, num_holdout_classes=3,
                            correlated_features=None,
                            restrict_train_sets_to=5000)
                            #restrict_train_sets_to=None)


def main_cst_fullset():
    save_dir = "/data2/lampinen/shared_structure/dataset_specs/cst_5000train"
    cst_stims_dir = "/data2/lampinen/color_texture_shape_stimuli"

    # file handling
    util.make_dirs([save_dir])

    multiversion_splits_cst(cst_stims_dir, save_dir, num_holdout_classes=3,
                            correlated_features=None,
                            restrict_train_sets_to=5000,
                            holdouts_on_target_attribute=False)


def main_cst_correlated():
    correlated_features = ["shape", "color"]
    min_train_set = None
    for cond_prob in [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        save_dir = "/data2/lampinen/shared_structure/dataset_specs/cst_correlated-{}_cond_prob-{}".format(
            "-".join(correlated_features), util.float_to_str(cond_prob))
        cst_stims_dir = "/data2/lampinen/color_texture_shape_stimuli"
        print("Creating: {}".format(save_dir))

        # file handling
        util.make_dirs([save_dir])

        this_train_set = multiversion_splits_cst(cst_stims_dir, save_dir, num_holdout_classes=3,
                                                 correlated_features=correlated_features, 
                                                 corr_feat_cond_prob=cond_prob,
                                                 restrict_train_sets_to=min_train_set)
        if cond_prob == 1.:
            min_train_set = this_train_set
    
if __name__=="__main__":
    #main_navon()
    # main_cst()
    main_cst_correlated()
    #main_cst_fullset()
