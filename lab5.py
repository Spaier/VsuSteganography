# 2. Реализовать алгоритм Cox. В качестве метрик для оценки искажений
# маркированных контейнеров использовать u maxD, uNC, uUQI.
# Оценить устойчивость встроенной информации по отношению негативному
# воздействию на заполненный контейнер типа аддитивного зашумления.

#%%
from typing import *
import random

from PIL import Image

from scipy.fftpack import dct, idct

from str_to_bit import *
from metrics import *

def round_pixel_value(pixel_value):
    pixel_value = round(pixel_value)
    if pixel_value > 255:
        pixel_value = 255
    elif pixel_value < 0:
        pixel_value = 0
    return pixel_value

def block_dct(image: Image, block_size, block_row, block_column, channel: int):
    channel_values = []
    for x in range(block_size * block_row, block_size * block_row + block_size):
        for y in range(block_size * block_column, block_size * block_column + block_size):
            channel_values.append(image.getpixel((x, y))[channel])
    return dct(channel_values)

def image_idct(dct, image: Image, block_size, block_row, block_column, channel: int):
    channel_values = idct(dct)
    i = 0
    width, height = image.size
    for x in range(block_size * block_row, block_size * block_row + block_size):
        for y in range(block_size * block_column, block_size * block_column + block_size):
            pixel = list(image.getpixel((x, y)))
            pixel[channel] = round_pixel_value(channel_values[i])
            image.putpixel((x, y), tuple(pixel))
            i += 1
    return image

def get_new_dct_value(dct_value, bit, power):
    return dct_value * (1 + power * (bit if bit == 1 else -1))

def cox_encode(image: Image, message_bits, power=2, block_size=8):
    image_with_message = image.copy()
    width, height = image.size
    blocks_row_count = height // block_size
    blocks_column_count = width // block_size
    for index, bit in enumerate(message_bits):
        block_row_index = index % blocks_row_count
        block_column_index = index // blocks_row_count
        dct_values = block_dct(image, block_size, block_row_index, block_column_index, 2)
        max_index = dct_values.argmax()
        dct_values[max_index] = get_new_dct_value(dct_values[max_index], bit, power)
        image_idct(dct_values, image_with_message, block_size, block_row_index, block_column_index, 2)
    return image_with_message

def cox_decode(image_original: Image, image_distorted: Image, message_bits_length, block_size=8):
    message_bits = []
    width, height = image_original.size
    blocks_row_count = height // block_size
    blocks_column_count = width // block_size
    for index in range(message_bits_length):
        block_row_index = index % blocks_row_count
        block_column_index = index // blocks_row_count
        dct_values1 = block_dct(image_original, block_size, block_row_index, block_column_index, 2)
        dct_values2 = block_dct(image_distorted, block_size, block_row_index, block_column_index, 2)
        max_index = dct_values1.argmax()
        dct_value1 = dct_values1[max_index]
        dct_value2 = dct_values2[max_index]
        if (dct_value2 > dct_value1):
            message_bits.append(1)
        else:
            message_bits.append(0)
    return message_bits

data = 'Hello There'
data_bits = tobits(data)
print(data_bits)

img_original = Image.open('in.png')

img_with_message = cox_encode(img_original, data_bits)
img_with_message.save('out5.png')

message = cox_decode(img_original, img_with_message, len(data_bits))
print(message)
print(frombits(message))

print(f'maxd = {maxd(img_original, img_with_message, 2)}')
print(f'nc = {nc(img_original, img_with_message, 2)}')
print(f'uqi = {uqi(img_original, img_with_message)}')

def add_noise(image, noise_power, channel: int):
    width, height = image.size
    pixel_count = width * height
    for x in range(width):
        for y in range(height):
            pixel = list(image.getpixel((x, y)))
            pixel[channel] = round_pixel_value(pixel[channel] + random.gauss(0, noise_power))
            image.putpixel((x, y), tuple(pixel))

noise_powers = [1, 10, 50, 100]
for noise_power in noise_powers:
    img_noise = img_with_message.copy()
    add_noise(img_noise, noise_power, 2)
    message = cox_decode(img_original, img_noise, len(data_bits))
    print(f'Noise applied - {noise_power}, Message = {frombits(message)}')
