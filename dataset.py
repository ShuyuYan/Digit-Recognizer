import torch
import clip
from PIL import Image
import csv
import numpy as np

data = []

with open('train.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    f = 0
    for row in reader:
        if f == 0:
            f = 1
            continue
        row_data = [int(value) for value in row]
        label = row_data[0]
        pixels = row_data[1:]
        img_array = np.array(pixels).reshape(28, 28).astype('uint8')
        img_name = "/home/yanshuyu/Data/DigitRecognizer/train/" + str(label) + "_" + str(f) + ".jpg"
        img = Image.fromarray(img_array)
        f += 1
        data.append(img_name)
        # img.save(img_name)

with open('train_path.txt', 'w') as f:
    for row in data:
        print(row, file=f)
