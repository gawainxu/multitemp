from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.ndimage
from PIL import Image

import os

BASE_SHAPES = ["triangle", "square", 
               "plus", "circle", "tee",
               "rhombus", "pentagon",
               "star", "fivesquare", "trapezoid"]

BASE_COLORS = {
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
    "blue": (0., 0., 1.),
    "yellow": (1., 1, 0.),
    "pink": (1., 0.4, 1.),
    "cyan": (0., 1., 1.),
    "purple": (0.3, 0., 0.5),
    "ocean": (0.1, 0.4, 0.5),
    "orange": (1., 0.6, 0.),
    "white": (1., 1., 1.),
}

BG_COLOR = np.array((0.5, 0.5, 0.5), dtype=np.float32)

BASE_TEXTURES =  ["solid", "stripes", "grid", "hexgrid", "dots", "noise",
                  "triangles", "zigzags", "rain", "pluses"]

BASE_SIZE = 128 

RENDER_SIZE = 224 

RANDOM_ANGLE_RANGE = 45

TEXTURE_SCALE = 10 

BASE_COLORS = {n: np.array(c, dtype=np.float32) for n, c in BASE_COLORS.items()} 

def make_dirs(dirs):
    # dirs is a list
    for d in dirs:
        if not os.path.isdir(d):
            os.makedirs(d)


def _render_plain_shape(name, size=BASE_SIZE):
    """Shape without color dimension."""
    size = int(size)
    shape = np.zeros([size, size], np.float32)
    if name == "square":
        shape[:, :] = 1.
    elif name == "circle":
        for i in range(size):
            for j in range(size):
                if np.square(i + 0.5 - size // 2) + np.square(j + 0.5 - size // 2) < np.square(size // 2):
                    shape[i, j] = 1.
    elif name == "triangle":
        for i in range(size):
            for j in range(size):
                if np.abs(j - size // 2) - np.abs(i // 2) < 1:
                    shape[i, j] = 1.
    elif name == "plus":
        shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 1.
        shape[size // 2 - size // 6: size // 2 + size //6 + 1, :] = 1.
    elif name == "tee":
        shape[:, size // 2 - size // 6: size // 2 + size //6 + 1] = 1.
        shape[:size // 3, :] = 1.
    elif name == "rhombus":
        for i in range(size):
            for j in range(size):
                if 0 < j - size // 2 + i // 2 < size // 2:
                    shape[i, j] = 1.
    elif name == "pentagon":
        midline = int(size * 0.4)
        for i in range(midline):
            for j in range(size):
                if np.abs(j - size // 2) - np.abs(i * 1.25) < 1:
                    shape[i, j] = 1.
        for i in range(midline, size):
            x_off = (i - midline) // 3.1
            for j in range(size):
                if  x_off < j < size - x_off:
                    shape[i, j] = 1.
    elif name == "star":
        line = int(size * 0.4)
        line2 = line + int(0.2 * size) 
        line3 = line + int(0.15 * size) 
        for i in range(line):
            for j in range(size):
                if np.abs(j - size // 2) - np.abs(i // 4) < 1:
                    shape[i, j] = 1.
        for i in range(line, line2):
            x_off = (i - line) * 2.4 
            for j in range(size):
                if  x_off < j < size - x_off:
                    shape[i, j] = 1.
        for i in range(line3, size):
            x_off_1 = (size * 0.33) - 0.43 * (i - line3) 
            x_off_2 = (size * 0.62) -  1.05 * (i - line3)
            for j in range(size):
                if  x_off_1 < j < x_off_2 or x_off_1 < size - j < x_off_2:
                    shape[i, j] = 1.
    elif name == "fivesquare":
        shape[:, :] = 1.
        shape[:, size // 3: 2 * size // 3] = 0.
        shape[size // 3: 2 * size // 3, :] = 0.
        shape[size // 3: 2 * size // 3, size // 3: 2 * size // 3] = 1.
    elif name == "trapezoid":
        for i in range(size):
            x_off = i // 3.1
            for j in range(size):
                if  x_off < j < size - x_off:
                    shape[i, j] = 1.

    return shape

def get_texture(size, texture_name):
    scale = TEXTURE_SCALE
    lwidth = scale // 3
    small_lwidth = scale // 5
    texture_offset_x = np.random.randint(0, scale)
    texture_offset_y = np.random.randint(0, scale)
    texture = np.zeros([size, size], dtype=np.float32)
    if texture_name == "solid":
        return np.ones_like(texture) 
    elif texture_name == "stripes":
        for i in range(size):
            if (i + texture_offset_y) % scale < lwidth:
                texture[i, :] = 1.
    elif texture_name == "grid":
        for i in range(size):
            if (i + texture_offset_y) % scale < lwidth:
                texture[i, :] = 1.
        for j in range(size):
            if (j + texture_offset_x) % scale < lwidth:
                texture[:, j] = 1.
    elif texture_name == "hexgrid":
        for i in range(size):
            for j in range(size):
                y = (i + texture_offset_y)
                x = (j + texture_offset_x)
                #if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                if (x + int(1.73 * y)) % scale < small_lwidth or (x - int(1.73 * y)) % scale < small_lwidth or y % scale < small_lwidth:
                    texture[i, j] = 1.
    elif texture_name == "dots":
        rad_squared = (3 * scale // 7)** 2 
        for i in range(size):
            for j in range(size):
                y = ((i + texture_offset_y) % scale) - scale // 2
                x = ((j + texture_offset_x) % scale) -  scale // 2
                #if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                if (x ** 2) + (y ** 2) < rad_squared:
                    texture[i, j] = 1.
    elif texture_name == "noise":
        texture = np.random.binomial(1, 0.5, texture.shape)
    elif texture_name == "triangles":
        for i in range(size):
            for j in range(size):
                y = (i + texture_offset_y) % scale
                x = (j + texture_offset_x) % scale
                #if y < lwidth or (x + int(1.73 * y)) % scale < lwidth:
                if  y // 2 - np.abs(x - scale // 2) > 0:
                    texture[i, j] = 1.
    elif texture_name == "zigzags":
        scale_off = scale - scale // 2 
        for i in range(size):
            slopesign = ((i + texture_offset_y) // scale) % 2 
            slopesign2 = ((i + texture_offset_y) //  (2 * scale)) % 2 
            for j in range(size):
                y = (i + texture_offset_y) % scale
                x = (j + texture_offset_x) % scale
                if slopesign:
                    x = scale - x - 1
                off = y // 2
                if  off < x < scale_off + off: 
                    texture[i, j] = 1.
    elif texture_name == "rain":
        rainheight = scale - scale // 3
        rainwidth = 1
        rainprob = 0.05
        this_offset_x = np.random.randint(0, scale)
        for i in range(size):
            for j in range(size):
                if np.random.binomial(1, rainprob):
                    texture[i: i + rainheight, j:j + rainwidth] = 1.
    elif texture_name == "pluses":
        pl_half_width = 1.5 
        for i in range(size):
            slopesign = ((i + texture_offset_y) // scale) % 2 
            for j in range(size):
                y = (i + texture_offset_y) % scale
                x = (j + texture_offset_x) % scale
                if slopesign:
                    if (np.abs(x) < pl_half_width) or (scale - x < pl_half_width) or ((np.abs(y - scale // 2) < pl_half_width) and np.abs(x - scale // 2) > pl_half_width): 
                    #if (np.abs(x - scale // 2) < pl_half_width and small_lwidth < y < scale - small_lwidth) or (np.abs(y - scale // 2) < pl_half_width and small_lwidth < x < scale - small_lwidth): 
                        texture[i, j] = 1.
                else:
                    if (np.abs(x - scale // 2) < pl_half_width) or (np.abs(y - scale // 2) < pl_half_width): 
                    #if (np.abs(x - scale // 2) < pl_half_width and small_lwidth < y < scale - small_lwidth) or (np.abs(y - scale // 2) < pl_half_width and small_lwidth < x < scale - small_lwidth): 
                        texture[i, j] = 1.

        
    return texture 


_base_templates = {(s, BASE_SIZE): _render_plain_shape(s, BASE_SIZE) for s in BASE_SHAPES} 


def render_uncolored_shape(name, size=BASE_SIZE):
    "Shape without color dimension, at random rotation and position."
    template = _base_templates[(name, size)]
    angle = np.random.randint(-RANDOM_ANGLE_RANGE, RANDOM_ANGLE_RANGE)
    shape = scipy.ndimage.rotate(template, angle, order=1)
    new_size = shape.shape
    image = np.zeros([RENDER_SIZE, RENDER_SIZE], np.float32)
    offset_x = np.random.randint(0, RENDER_SIZE - new_size[0])
    offset_y = np.random.randint(0, RENDER_SIZE - new_size[1])
    image[offset_x:offset_x + new_size[0], 
          offset_y:offset_y + new_size[1]] = shape
    return image 


def render_stimulus(shape, color, texture):
    image = render_uncolored_shape(shape)
    t_size =  2 * RENDER_SIZE
    texture = get_texture(t_size, texture) 
    angle = np.random.randint(-RANDOM_ANGLE_RANGE, RANDOM_ANGLE_RANGE)
    texture = scipy.ndimage.rotate(texture, angle, order=0, reshape=False)
    texture = texture[RENDER_SIZE//2:-RENDER_SIZE//2, RENDER_SIZE//2:-RENDER_SIZE//2]
    image = np.multiply(image, texture)
    color_image = image[:, :, None] * BASE_COLORS[color][None, None, :]
    color_image +=  (1-image)[:, :, None] * BG_COLOR[None, None, :]

    return color_image
    

def save_stimuli(output_directory="./color_texture_shape_stimuli/", 
                 num_per_combination=10):
    """Saves stimuli where all features vary orthogonally."""
    make_dirs([output_directory])
    for s in BASE_SHAPES:
        for t in BASE_TEXTURES:
            for c in BASE_COLORS.keys():
                print(s, t, c)
                for i in range(num_per_combination):
                    image_array = render_stimulus(s, c, t)
                    image = Image.fromarray((image_array * 255.).astype(np.uint8), mode='RGB')
                    image.save(output_directory + "%s_%s_%s_%i.png" % (s, t, c, i))


if __name__ == "__main__":
    save_stimuli("/data2/lampinen/color_texture_shape_stimuli/")





