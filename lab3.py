# 4. Реализовать алгоритм Брайн докса. В качестве метрик для оценки
# искажений заполненных контейнеров использовать uPSNR, uMSE, uUQI.
# Построить зависимости вероятности ошибок при извлечении скрытых данных от параметра
# d E N (шаг переквантования)

#%%
from typing import *
import random

from PIL import Image, ImageChops

import numpy as np

from str_to_bit import *
from metrics import *

luminosity_coeffs = [0.299, 0.587, 0.144]

def get_luminosity(pixel):
    return luminosity_coeffs[0] * pixel[0] + luminosity_coeffs[1] * pixel[1] + luminosity_coeffs[2] * pixel[2]

def get_masks(block_size, mask_count):
    random.seed(555)
    masks = []
    for mask_index in range(mask_count):
        mask = [[0] * block_size for i in range(block_size)]
        for x in range(block_size):
            for y in range(block_size):
                mask[x][y] = random.randrange(0, 2, 1)
        masks.append(mask)
    return masks

def change_brightness(image: Image, pixels, luminosity_change):
    luminosity_error = 0
    for x, y, luminosity in pixels:
        pixel = list(image.getpixel((x, y)))
        luminosity_changed = 0
        for channel in range(3):
            if (abs(luminosity_changed) >= abs(luminosity_change)):
                break
            luminosity_coeff = luminosity_coeffs[channel]
            if (luminosity_change > 0 and pixel[channel] < 255 or luminosity_change < 0 and pixel[channel] > 0):
                channel_change = round((luminosity_change - luminosity_changed) / luminosity_coeff)
                pixel[channel] += channel_change
                if (pixel[channel] > 255):
                    channel_change -= pixel[channel] - 255
                    pixel[channel] = 255
                elif (pixel[channel] < 0):
                    channel_change -= pixel[channel]
                    pixel[channel] = 0
                luminosity_changed += luminosity_coeff * channel_change
        luminosity_error += luminosity_changed - luminosity_change
        image.putpixel((x, y), tuple(pixel))
    luminosity_error /= len(pixels)
    return luminosity_error

def get_luminosity_halves(block_pixels, block_size, height, luminosity_change, luminosity_threshold):
    sorted_pixels = sorted(block_pixels, key=lambda p: p[2])

    brightHalf = sorted(sorted_pixels[block_size ** 2 // 2:], key=lambda p: p[1] * height + p[0])
    darkHalf = sorted(sorted_pixels[:block_size ** 2 // 2], key=lambda p: p[1] * height + p[0])

    brightMean = sum(map(lambda x: x[2], brightHalf)) / len(brightHalf)
    darkMean = sum(map(lambda x: x[2], darkHalf)) / len(darkHalf)

    if (brightMean - darkMean < luminosity_threshold):
        return False, [], []

    brightMinPixel = min(brightHalf, key=lambda x: x[2])
    darkMaxPixel = max(darkHalf, key=lambda x: x[2])
    i = 0
    while True:
        if i % 2 == 0:
            brightMinPixel = min(brightHalf, key=lambda x: x[2])
            if darkMaxPixel[2] + (luminosity_change / len(darkHalf)) >= brightMinPixel[2] - (luminosity_change / len(brightHalf)):
                brightHalf.remove(brightMinPixel)
            else:
                break
        else:
            darkMaxPixel = max(darkHalf, key=lambda x: x[2])
            if darkMaxPixel[2] + (luminosity_change / len(darkHalf)) >= brightMinPixel[2] - (luminosity_change / len(brightHalf)):
                darkHalf.remove(darkMaxPixel)
            else:
                break
    return True, brightHalf, darkHalf

def get_groups(luminosity_half, mask, block_row, block_column, block_size):
    halfA = []
    halfB = []
    for index, pixel in enumerate(luminosity_half):
        x, y, l = pixel
        mask_value = mask[x - block_row * block_size][y - block_column * block_size]
        if mask_value == 0:
            halfA.append(pixel)
        elif mask_value == 1:
            halfB.append(pixel)
    return halfA, halfB

def bruyndonckx_encode(image: Image, message_bits, block_size=8, reQuantization=1, luminosity_change=1, luminosity_threshold=1):
    image_with_message = image.copy()
    width, height = image.size
    blocks_row_count = height // block_size
    blocks_column_count = width // block_size
    blocks_count = blocks_row_count * blocks_column_count
    masks = get_masks(block_size, len(message_bits) * 2)
    bit_index = 0
    for block_index in range(blocks_count):
        if (bit_index >= len(message_bits)):
            break

        bit = message_bits[bit_index]
        block_pixels = []
        block_row_index = block_index % blocks_row_count
        block_column_index = block_index // blocks_row_count
        for x in range(block_row_index * block_size, block_row_index * block_size + block_size):
            for y in range(block_column_index * block_size, block_column_index * block_size + block_size):
                pixel = image.getpixel((x, y))
                luminosity = get_luminosity(pixel)
                block_pixels.append((x, y, luminosity))
        takeBlock, brightHalf, darkHalf = get_luminosity_halves(block_pixels, block_size, height, luminosity_change, luminosity_threshold)

        if not takeBlock:
            continue

        brightMask = masks[bit_index * 2]
        darkMask = masks[bit_index * 2 + 1]
        brightHalfA, brightHalfB = get_groups(brightHalf, brightMask, block_row_index, block_column_index, block_size)
        darkHalfA, darkHalfB = get_groups(darkHalf, darkMask, block_row_index, block_column_index, block_size)
        brightMeanA = sum(map(lambda x: x[2], brightHalfA)) / len(brightHalfA)
        brightMeanB = sum(map(lambda x: x[2], brightHalfB)) / len(brightHalfB)
        darkMeanA = sum(map(lambda x: x[2], darkHalfA)) / len(darkHalfA)
        darkMeanB = sum(map(lambda x: x[2], darkHalfB)) / len(darkHalfB)

        if bit == 1:
            a_change = +1
            b_change = -1
        else:
            a_change = -1
            b_change = +1

        updateGroupValue = lambda group: list(map(lambda p: (p[0], p[1], get_luminosity(image_with_message.getpixel((p[0], p[1])))), group))

        reQuantization_brightA = reQuantization * len(brightHalfA) / (len(brightHalfA) + len(brightHalfB))
        reQuantization_brightB = reQuantization * len(brightHalfB) / (len(brightHalfA) + len(brightHalfB))

        brightMean = sum(map(lambda x: x[2], brightHalf)) / len(brightHalf)
        darkMean = sum(map(lambda x: x[2], darkHalf)) / len(darkHalf)

        while (brightMeanA < brightMeanB + luminosity_change and bit == 1) or (brightMeanA + luminosity_change > brightMeanB and bit == 0):
            a_error = change_brightness(image_with_message, brightHalfA, a_change * reQuantization_brightA)
            change_brightness(image_with_message, brightHalfB, b_change * (reQuantization_brightB + a_error))
            brightHalfA = updateGroupValue(brightHalfA)
            brightHalfB = updateGroupValue(brightHalfB)
            brightMeanA = sum(map(lambda x: x[2], brightHalfA)) / len(brightHalfA)
            brightMeanB = sum(map(lambda x: x[2], brightHalfB)) / len(brightHalfB)

        reQuantization_darkA = reQuantization * len(darkHalfA) / (len(darkHalfA) + len(darkHalfB))
        reQuantization_darkB = reQuantization * len(darkHalfB) / (len(darkHalfA) + len(darkHalfB))
        while (darkMeanA < darkMeanB + luminosity_change and bit == 1) or (darkMeanA + luminosity_change > darkMeanB and bit == 0):
            a_error = change_brightness(image_with_message, darkHalfA, a_change * reQuantization_darkA)
            change_brightness(image_with_message, darkHalfB, b_change * (reQuantization_darkB + a_error))
            darkHalfA = updateGroupValue(darkHalfA)
            darkHalfB = updateGroupValue(darkHalfB)
            darkMeanA = sum(map(lambda x: x[2], darkHalfA)) / len(darkHalfA)
            darkMeanB = sum(map(lambda x: x[2], darkHalfB)) / len(darkHalfB)

        brightHalf = updateGroupValue(brightHalf)
        darkHalf = updateGroupValue(darkHalf)
        brightMeanNew = sum(map(lambda x: x[2], brightHalf)) / len(brightHalf)
        darkMeanNew = sum(map(lambda x: x[2], darkHalf)) / len(darkHalf)
        if (brightMeanNew - darkMeanNew < luminosity_threshold):
            raise Exception(f'Luminosity broken. New={darkMeanNew, brightMeanNew}')

        bit_index += 1

    return image_with_message

def bruyndonckx_decode(image_distorted: Image, message_bits_length, block_size=8, reQuantization=1, luminosity_change=1, luminosity_threshold=1):
    width, height = image_distorted.size
    blocks_row_count = height // block_size
    blocks_column_count = width // block_size
    blocks_count = blocks_row_count * blocks_column_count
    masks = get_masks(block_size, message_bits_length * 2)
    message_bits = []
    bit_index = 0
    for block_index in range(blocks_count):
        if (bit_index >= message_bits_length):
            break
        block_pixels = []
        block_row_index = block_index % blocks_row_count
        block_column_index = block_index // blocks_row_count
        for x in range(block_row_index * block_size, block_row_index * block_size + block_size):
            for y in range(block_column_index * block_size, block_column_index * block_size + block_size):
                pixel = image_distorted.getpixel((x, y))
                luminosity = get_luminosity(pixel)
                block_pixels.append((x, y, luminosity))
        takeBlock, brightHalf, darkHalf = get_luminosity_halves(block_pixels, block_size, height, luminosity_change, luminosity_threshold)
        if not takeBlock:
            continue

        brightMask = masks[bit_index * 2]
        darkMask = masks[bit_index * 2 + 1]
        brightHalfA, brightHalfB = get_groups(brightHalf, brightMask, block_row_index, block_column_index, block_size)
        darkHalfA, darkHalfB = get_groups(darkHalf, darkMask, block_row_index, block_column_index, block_size)
        brightMeanA = sum(map(lambda x: x[2], brightHalfA)) / len(brightHalfA)
        brightMeanB = sum(map(lambda x: x[2], brightHalfB)) / len(brightHalfB)
        darkMeanA = sum(map(lambda x: x[2], darkHalfA)) / len(darkHalfA)
        darkMeanB = sum(map(lambda x: x[2], darkHalfB)) / len(darkHalfB)
        if (darkMeanA + luminosity_change < darkMeanB and brightMeanA + luminosity_change < brightMeanB):
            message_bits.append(0)
        elif (darkMeanA + luminosity_change > darkMeanB and brightMeanA + luminosity_change > brightMeanB):
            message_bits.append(1)
        bit_index += 1
    return message_bits

# data = 'Hi'
# data_bits = tobits(data)
data_bits = [0, 1, 0, 1]
print(data_bits)

img_original = Image.open('in.png')

l = 5
l_t = 10
d = 10
b = 4

img_with_message = bruyndonckx_encode(img_original, data_bits, block_size=b, luminosity_change=l, luminosity_threshold=l_t, reQuantization=d)
img_with_message.save('out3.png')

message = bruyndonckx_decode(img_with_message, len(data_bits), block_size=b, luminosity_change=l, luminosity_threshold=l_t, reQuantization=d)
print(message)
print(frombits(message))

print(f'psnr = {psnr(img_original, img_with_message, 2)}')
print(f'mse = {mse(img_original, img_with_message, 2)}')
uqi_result = uqi(img_original, img_with_message)
print(f'uqi = {uqi_result} {uqi_result >= -1} {uqi_result <= 1}')

reQuantizationValues = [0.5, 1, 2, 5, 10]
messages = []
for reQuantization in reQuantizationValues:
    img_with_message = bruyndonckx_encode(img_original, data_bits, block_size=b, luminosity_change=l, luminosity_threshold=l_t, reQuantization=reQuantization)
    message = bruyndonckx_decode(img_with_message, len(data_bits), block_size=b, luminosity_change=l, luminosity_threshold=l_t, reQuantization=reQuantization)
    messages.append(frombits(message))
    print(f're-quantization={reQuantization}, bits decoded={message}')
#%%
