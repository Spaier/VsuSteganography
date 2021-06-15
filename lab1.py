# 8) Реализовать LSB-алгоритм. В качестве метрик для оценки искажений
# заполненных контейнеров использовать μNMSE, μMSE, μLMSE
# При скрытии использовать псевдослучайный порядок обхода пикселей контейнера при реализации ССИ.

#%%
from typing import *
import random

from PIL import Image, ImageChops

import numpy as np

from str_to_bit import *
from metrics import *

def lsb_encode(image: Image, message: str):
    image_with_message = image.copy()
    message_bits = tobits(message)
    width, height = image.size
    pixel_amount = width * height
    pixel_indices = range(0, pixel_amount)
    random.seed(777)
    pixel_values = random.sample(pixel_indices, k=pixel_amount)
    for index, pixel_value in enumerate(pixel_values):
        if (len(message_bits) > index):
            bit = message_bits[index]
            pixel_x = pixel_value % width
            pixel_y = pixel_value // width
            pixel = list(image_with_message.getpixel((pixel_x, pixel_y)))
            pixel[0] = pixel[0] & ~1 | int(bit)
            image_with_message.putpixel((pixel_x, pixel_y), tuple(pixel))
    return image_with_message

def lsb_decode(image_original, image_distorted) -> str:
    width, height = image_original.size
    pixel_amounts = width * height
    samples = range(0, pixel_amounts)
    random.seed(777)
    pixel_values = random.sample(samples, k=pixel_amounts)
    message_bits = []
    for index, pixel_value in enumerate(pixel_values):
        pixel_x = pixel_value % width
        pixel_y = pixel_value // width
        pixel = list(image_distorted.getpixel((pixel_x, pixel_y)))
        bit = pixel[0] & 1
        message_bits.append(bit)
    message = frombits(message_bits)
    return message

data = 'Hi'

img_original = Image.open('in.png')

img_with_message = lsb_encode(img_original, data)
img_with_message.save('out1.png')

message = lsb_decode(img_original, img_with_message)
print(message[:len(data)])

print(f'nmse = {nmse(img_original, img_with_message, 0)}')
print(f'mse = {mse(img_original, img_with_message)}')
print(f'lmse = {lmse(img_original, img_with_message)}')
