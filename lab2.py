# 6) Реализовать метод Куттера-Джордана-Боссена. В качестве метрик
# для оценки искажений заполненных контейнеров использовать
# u_maxD, u_SNR, u_PSNR.
# Оценить устойчивость встроенной информации по отношению
# негативному воздействию на заполненный контейнер типа JPEG-компрессии
# с различными значениями показателя качества сжатия

#%%
from typing import *
import random

from PIL import Image, ImageChops

import numpy as np

from str_to_bit import *
from metrics import *

blue_channel_id = 2

def kdb_encode(image: Image, message_bits, luminosity_change, encode_per_bit=5):
    image_with_message = image.copy()
    width, height = image.size
    pixel_amount = width * height
    pixel_indices = range(0, pixel_amount)
    random.seed(777)
    pixel_values = random.sample(pixel_indices, k=pixel_amount)
    message_bit_index = 0
    pixel_index = 0
    for pixel_value in pixel_values:
        bit = message_bits[message_bit_index]
        pixel_x = pixel_value % width
        pixel_y = pixel_value // width
        pixel = list(image_with_message.getpixel((pixel_x, pixel_y)))
        old_value = pixel[blue_channel_id]
        pixel[blue_channel_id] = new_blue(image, pixel_x, pixel_y, bit, luminosity_change)
        image_with_message.putpixel((pixel_x, pixel_y), tuple(pixel))
        message_bit_index += 1
        pixel_index += 1
        if message_bit_index % len(message_bits) == 0:
            message_bit_index = 0
        if pixel_index > len(message_bits) * encode_per_bit:
            break

    return image_with_message

def kdb_decode(image_original, image_distorted, message_length, cross_distance=3, encode_per_bit=5) -> str:
    width, height = image_original.size
    pixel_amount = width * height
    pixel_indices = range(0, pixel_amount)
    random.seed(777)
    pixel_values = random.sample(pixel_indices, k=pixel_amount)
    messages = []
    message_bits = []
    message_bit_index = 0
    pixel_index = 0
    for pixel_value in pixel_values:
        pixel_x = pixel_value % width
        pixel_y = pixel_value // width
        bit = get_from_blue(image_distorted, pixel_x, pixel_y, cross_distance)
        message_bits.append(bit)
        message_bit_index += 1
        pixel_index += 1
        if message_bit_index % message_length == 0:
            message_bit_index = 0
            messages.append(frombits(message_bits))
            message_bits = []
        if pixel_index > message_length * encode_per_bit:
            break
    return messages

def get_luminosity(r, g, b, luminosity_change):
    luminosity = 0.299 * r + 0.587 * g + 0.144 * b
    return 5 / luminosity_change if luminosity == 0 else luminosity

def new_blue(image: Image, w, h, bit, luminosity_change):
    pixel = image.getpixel((w, h))
    new_blue_value = pixel[blue_channel_id] + (2 * bit - 1) * get_luminosity(pixel[0], pixel[1], pixel[blue_channel_id], luminosity_change) * luminosity_change
    if new_blue_value < 0:
        new_blue_value = 0
    elif new_blue_value > 255:
        new_blue_value = 255
    return round(new_blue_value)

def get_from_blue(image: Image, w, h, cross_distance):
    width, height = image.size
    blue_prediction = 0
    for i in range(1, cross_distance):
        if h + i < height and h + i >= 0:
            pixel = image.getpixel((w, h + i))
            blue_prediction += pixel[blue_channel_id]
        if h - i < height and h - i >= 0:
            pixel = image.getpixel((w, h - i))
            blue_prediction += pixel[blue_channel_id]
        if w + i < width and w + i >= 0:
            pixel = image.getpixel((w + i, h))
            blue_prediction += pixel[blue_channel_id]
        if w - i < width and w - i >= 0:
            pixel = image.getpixel((w - i, h))
            blue_prediction += pixel[blue_channel_id]
    pixel = image.getpixel((w, h))
    blue_prediction /= (4 * cross_distance)
    bit = 1 if pixel[blue_channel_id] - blue_prediction > 0 else 0
    return bit

data = 'Hi'

img_original = Image.open('in.png')

data_bits = tobits(data)

img_with_message = kdb_encode(img_original, data_bits, 0.4)
img_with_message.save('out2.png')

messages = kdb_decode(img_original, img_with_message, len(data_bits))
print(messages)

print(f'maxd = {maxd(img_original, img_with_message, blue_channel_id)}')
print(f'snr = {snr(img_original, img_with_message, blue_channel_id)}')
print(f'psnr = {psnr(img_original, img_with_message, blue_channel_id)}')

qualities = [0, 25, 50, 75, 95]
for quality in qualities:
    img_with_message.save(f'out2_{quality}.jpeg', 'JPEG', optimize = True, quality = quality)
    messages = kdb_decode(img_original, img_with_message, len(data_bits))
    print(f'Quality - {quality}')
    print(messages)
