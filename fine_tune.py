import os
from PIL import Image
import numpy as np
import clip
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

class YourDataset(Dataset):
    def __init__(self, img_root, meta_root, is_train, preprocess):
        self.img_root = img_root
        self.meta_root = meta_root
        self.train_set_file = os.path.join(meta_root, 'train_path.txt')
        self.test_set_file = os.path.join(meta_root, 'test_path.txt')
        self.is_train = is_train
        self.img_process = preprocess
        self.samples = []
        self.sam_labels = []
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file

        with open(self.read_file, 'r') as f:
            for line in f:
                img_path = line[:-1]
                label = line.split('/')[-1]
                digit = label[0]
                self.samples.append(img_path)
                self.sam_labels.append(digit)

        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.img_process(image)
        return image, token


device = "cuda" if torch.cuda.is_available() else "cpu"
net, preprocess = clip.load("ViT-B/32", device=device, jit=False)

optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
your_dataset = YourDataset(img_root='/train', meta_root='/home/yanshuyu/Data/DigitRecognizer', is_train=True, preprocess=preprocess)
dataset_size = len(your_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)

phase = "train"
model_name = "Digit"
ckt_gap = 10
for epoch in range(1, 21):
    scheduler.step()
    total_loss = 0
    batch_num = 0
    with torch.cuda.amp.autocast(enabled=True):
        for images,label_tokens in your_dataloader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2
                total_loss += cur_loss
                if phase == "train":
                    cur_loss.backward()
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        clip.model.convert_weights(net)
            if batch_num % 4 == 0:
                logger.info('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
        epoch_loss = total_loss / dataset_size
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f"{model_name}_epoch_{epoch}_v1.pth")
        logger.info(f"weights_{epoch} saved")
        if epoch % ckt_gap == 0:
            checkpoint_path = f"{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
        logger.info('{} Loss: {:.4f}'.format(
            phase, epoch_loss))
