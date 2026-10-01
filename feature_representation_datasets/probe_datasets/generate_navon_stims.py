import os
import sys
sys.path.append("../")

import string
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import random

import util
import pdb


"""Generates Navon dataset.

Modified from Navon dataset used in Hermann et al. 2019 (rotate textures 
independently of shape), based on stimuli developed by Navon 1977.
"""

def make_dirs(dirs):
    # dirs is a list
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)


def polar_to_cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def letter_to_shifted_masks(letter, im_size=224, font_size=200,
                            font_path="/Library/Fonts/Arial/Arial Bold.ttf"):
    image = Image.new("RGB", (im_size, im_size), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, font_size)
    w, h = draw.textsize(letter, font=font)
    centered_x, centered_y = ((im_size - w)/2, 0) 
    
    masks = []
    for i, r in enumerate(np.linspace(0, 40, num=6)[1:]): # norm r
        phi = np.random.uniform(0, 2*math.pi)
        shift_x, shift_y = map(int, map(round, polar_to_cartesian(r, phi)))
        print("r = {}, shift = ({}, {})".format(r, shift_x, shift_y))
        x, y = centered_x + shift_x, centered_y + shift_y
        image = Image.new("RGB", (im_size, im_size), "white")
        draw = ImageDraw.Draw(image)
        draw.text((x, y), letter, font=font, fill="black")
        masks.append(image)
    return masks


def render_twice_rotated(mask, fill_letter, savename, savedir, font_size=8,
                         font_path="/Library/Fonts/Arial/Arial Bold.ttf"):
    mask = np.array(mask)
    image = Image.new("RGB", (mask.shape[0], mask.shape[1]), "white")
    draw = ImageDraw.Draw(image)

    # Randomly sample a rotation for the large letter.
    rotation_deg = random.sample(range(-45, 46), k=1)[0]
    # and for the texture
    texture_rotation_deg = random.sample(range(-45, 46), k=1)[0]

    # Rotate the mask appropriately.
    PIL_mask = Image.fromarray(mask)
    PIL_mask = PIL_mask.rotate(rotation_deg, fillcolor="white", resample=Image.NEAREST)    
    mask = np.array(PIL_mask)

    # Generate the texture patch (small letter) that will repeat.
    font = ImageFont.truetype(font_path, font_size)
    step_size  = max(draw.textsize(fill_letter, font=font))

    texture = Image.new("RGB", (step_size, step_size), "white")
    texture_draw = ImageDraw.Draw(texture)
    texture_draw.text((0, 0), fill_letter, font=font, fill="black")
    texture = texture.rotate(rotation_deg + texture_rotation_deg, expand=1, resample=Image.BILINEAR, fillcolor="white")
    pastemask = Image.fromarray(np.array(texture)[:, :, 0] < 255)

    # Generate the textured image.
    for x in np.arange(0, mask.shape[0], step_size):
        for y in np.arange(0, mask.shape[1], step_size):
            if np.array_equal(mask[x, y, :], [0, 0, 0]):
                image.paste(texture, (y, x), mask=pastemask)

    image.save(os.path.join(savedir, "{}.png".format(savename)))


def render(mask, fill_letter, savename, savedir, font_size=8,
           font_path="/Library/Fonts/Arial/Arial Bold.ttf"):
    mask = np.array(mask)
    image = Image.new("RGB", (mask.shape[0], mask.shape[1]), "white")
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_path, font_size)
    step_size  = max(draw.textsize(fill_letter, font=font))

    for x in np.arange(0, mask.shape[0], step_size):
        for y in np.arange(0, mask.shape[1], step_size):
            if np.array_equal(mask[x, y, :], [0, 0, 0]):
                draw.text((y, x), fill_letter, font=font, fill="black")

    # Randomly sample a rotation on the fly.
    rotation_deg = random.sample(range(-45, 46), k=1)[0]

    image = image.rotate(rotation_deg, fillcolor="white", resample=Image.BILINEAR)    
    image.save(os.path.join(savedir, "{}.png".format(savename)))


def make_stims(savedir="navon_stims", twice_rotated=False,
               font_path="/Library/Fonts/Arial/Arial Bold.ttf"):
    util.make_dirs([savedir])

    all_letters = list(string.ascii_uppercase)
    util.make_dirs([os.path.join(savedir, shape_letter) 
                    for shape_letter in all_letters]) 

    for fill_letter in all_letters:
        for shape_letter in all_letters:
            masks = letter_to_shifted_masks(shape_letter, font_path=font_path)
            for i, mask in enumerate(masks):
                savename = "{}_{}-{}".format(shape_letter, i, fill_letter)
                if twice_rotated:
                    render_twice_rotated(mask, fill_letter, savename, 
                                         os.path.join(savedir, shape_letter),
                                         font_path=font_path)

                else:
                    render(mask, fill_letter, savename, 
                           os.path.join(savedir, shape_letter))


if __name__=="__main__":
    make_stims(savedir="navon_twice_rotated", twice_rotated=True,
               font_path="/Library/Fonts/Arial/Arial Bold.ttf")


