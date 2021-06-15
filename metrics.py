from typing import *

import numpy as np
import math

from PIL import Image, ImageChops

def nmse(image1: Image, image2: Image, channel: int):
    image_diff = ImageChops.difference(image1, image2)
    width, height = image1.size
    channel_metrics = []
    for n in range(0,3):
        summation = 0.0
        for x in range(0, width):
            for y in range(0, height):
                pixel = list(image_diff.getpixel((x, y)))
                summation += pixel[n]**2
        divide_by = 0.0
        for x in range(0, width):
            for y in range(0, height):
                pixel = list(image1.getpixel((x, y)))
                divide_by += pixel[n]**2
        metric = summation / divide_by
        channel_metrics.append(metric)
    return channel_metrics[channel]

def mse(image1: Image, image2: Image, channel: int):
    image_diff = ImageChops.difference(image1, image2)
    width, height = image1.size

    channel_metrics = []
    for n in range(0,3):
        summation = 0.0
        for x in range(0, width):
            for y in range(0, height):
                pixel = list(image_diff.getpixel((x, y)))
                summation += pixel[n]**2
        summation /= (width * height)
        channel_metrics.append(summation)
    return channel_metrics[channel]

def laplas(img: Image, x, y, n):
    pixel = list(img.getpixel((x, y)))
    pixels_to_check = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 1)
    ]
    minus_count = 0
    summation = 0.0
    for pixel_to_check in pixels_to_check:
        nearby_x = x + pixel_to_check[0]
        nearby_y = y + pixel_to_check[1]
        if nearby_x >= 0 and nearby_x < img.width and nearby_y >= 0 and nearby_y < img.height:
            nearby_pixel = list(img.getpixel((nearby_x, nearby_y)))
            summation += nearby_pixel[n]
            minus_count += 1
    summation -= pixel[n] * minus_count
    return summation

def lmse(image1: Image, image2: Image):
    width, height = image1.size

    channel_metrics = []
    for n in range(0,3):
        summation = 0.0
        for x in range(0, width):
            for y in range(0, height):
                summation += (laplas(image1, x, y, n) - laplas(image2, x, y, n))**2
        divide_by = 0.0
        for x in range(0, width):
            for y in range(0, height):
                divide_by += laplas(image1, x, y, n)**2
        metric = summation / divide_by
        channel_metrics.append(metric)
    return channel_metrics[0]

def maxd(image1: Image, image2: Image, channel: int):
    image_diff = ImageChops.difference(image1, image2)
    width, height = image1.size

    channel_metrics = []
    for n in range(0,3):
        max_d = 0
        for x in range(0, width):
            for y in range(0, height):
                pixel = list(image_diff.getpixel((x, y)))
                if pixel[n] > max_d:
                    max_d = pixel[n]
        channel_metrics.append(max_d)
    return channel_metrics[channel]

def lp_norm(image1: Image, image2: Image, p: int, channel: int):
    image_diff = ImageChops.difference(image1, image2)
    width, height = image1.size

    channel_metrics = []
    for n in range(0,3):
        max_d = 0
        for x in range(0, width):
            for y in range(0, height):
                pixel = list(image_diff.getpixel((x, y)))
                if pixel[n] > max_d:
                    max_d = pixel[n]
        channel_metrics.append(max_d)
    return channel_metrics[channel]

def snr(image1: Image, image2: Image, channel: int):
    return 1 / nmse(image1, image2, channel)

def psnr(image1: Image, image2: Image, channel: int):
    image_diff = ImageChops.difference(image1, image2)
    width, height = image1.size
    max_channel_value = 0
    diff_sum = 0
    for x in range(0, width):
        for y in range(0, height):
            pixel = list(image1.getpixel((x, y)))
            if pixel[channel] > max_channel_value:
                max_channel_value = pixel[channel]

            pixel = list(image_diff.getpixel((x, y)))
            diff_sum += pixel[channel]
    if diff_sum == 0:
        return 0
    result = width * height * max_channel_value ** 2 / diff_sum
    return result

def uqi(image1: Image, image2: Image):
    width, height = image1.size
    pixel_amount = width * height
    uqis = []
    for channel in range(0,3):
        mean_1 = 0
        mean_2 = 0
        for x in range(0, width):
            for y in range(0, height):
                mean_1 += image1.getpixel((x, y))[channel]
                mean_2 += image2.getpixel((x, y))[channel]
        mean_1 /= pixel_amount
        mean_2 /= pixel_amount
        variance_1 = 0
        variance_2 = 0
        for x in range(0, width):
            for y in range(0, height):
                variance_1 += (image1.getpixel((x, y))[channel] - mean_1) ** 2
                variance_2 += (image2.getpixel((x, y))[channel] - mean_2) ** 2
        variance_1 /= pixel_amount
        variance_2 /= pixel_amount
        correlation = 0
        for x in range(0, width):
            for y in range(0, height):
                correlation += (image1.getpixel((x, y))[channel] - mean_1) * (image2.getpixel((x, y))[channel] - mean_2)
        correlation /= pixel_amount
        channel_uqi = 4 * correlation * mean_1 * mean_2 / ((variance_1 + variance_2) * (mean_1 ** 2 + mean_2 ** 2))
        uqis.append(channel_uqi)
    return np.prod(uqis) ** (1/3)

def nc(image1: Image, image2: Image, channel: int):
    width, height = image1.size
    pixel_amount = width * height
    image_diff_sum = 0
    image_square_sum = 0
    for x in range(0, width):
        for y in range(0, height):
            image1_pixel_value = image1.getpixel((x, y))[channel]
            image2_pixel_value = image2.getpixel((x, y))[channel]
            image_diff_sum += image1_pixel_value * image2_pixel_value
            image_square_sum += image1_pixel_value * image1_pixel_value
    return image_diff_sum  / image_square_sum
