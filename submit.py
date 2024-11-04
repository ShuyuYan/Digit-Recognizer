import torch
import clip
from PIL import Image
import csv
import numpy as np


ans = []
labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
weight_path = './Digit_epoch_20_v1.pth'
model.load_state_dict(torch.load(weight_path, map_location=device))
text = clip.tokenize(labels).to(device)

with open('test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    id = 0
    for row in reader:
        if id == 0:
            ans.append(["ImageId", "Label"])
            id = 1
            continue
        row_data = [int(value) for value in row]
        img_array = np.array(row_data).reshape(28, 28).astype('uint8')
        img = Image.fromarray(img_array)
        image = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_prob_index = probs.argmax()
        max_prob_label = labels[max_prob_index]
        ans.append([id, max_prob_label[-1]])
        id += 1

with open('submission.csv', 'w') as f:
    for i in ans:
        print(f'{i[0]},{i[1]}', file=f)
