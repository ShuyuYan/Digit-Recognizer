import torch
import clip
from PIL import Image
import csv
import numpy as np

data = []
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
weight_path = './Digit_epoch_20_v1.pth'
model.load_state_dict(torch.load(weight_path, map_location=device))
text = clip.tokenize(labels).to(device)
tot, right = 0, 0

with open('test_path.txt', 'r') as f:
    for row in f:
        tot += 1
        img_path = row[:-1]
        label = row.split('/')[-1]
        digit = label[0]
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print("Label probs:", probs)
        max_prob_index = probs.argmax()
        max_prob_label = labels[max_prob_index][-1]
        # print("Max probability label:", max_prob_label)
        right += str(digit) == str(max_prob_label)

print(right/tot)
