import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel
import json
from PIL import Image
import os
from tqdm import tqdm
from torchvision import transforms


# 1. 定义自定义数据集类
class JsonImageTextDataset(Dataset):
    def __init__(self, json_file, image_dir, processor,is_train, max_length=128):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.is_train=is_train

        # 读取JSON文件
        with open(json_file, 'r') as f:
            self.samples = json.load(f)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample["image"]
        caption = sample["caption"]
        image_path = os.path.join(self.image_dir, image_id)

        try:
            image = Image.open(image_path).convert('RGB')
            # image = train_transform(image) if self.is_train else valid_transform(image)
            # # 将张量转换回PIL图像（因为processor需要PIL输入）
            # image = transforms.ToPILImage()(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 创建空白图像时也应用变换
            image = Image.new('RGB', (256, 256))
            # image = train_transform(image) if self.is_train else valid_transform(image)
            # image = transforms.ToPILImage()(image)

        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

        inputs["labels"] = inputs["input_ids"].clone()
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


# 2. 定义生成函数
def generate_captions(model, processor, image_file, image_dir, output_file):
    with open(image_file, 'r', encoding='utf-8') as f:
        datas = f.readlines()

    all_set = []
    for data in tqdm(datas):
        sample_dict = {}
        img = data.rstrip('\n')
        image_path = os.path.join(image_dir, img)
        try:
            image = Image.open(image_path).convert('RGB')

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")

            image = Image.new('RGB', (256, 256))


        inputs = processor(images=image, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # print("\r")
        # print(f"image_path: {image_path}")
        # print(f"Generated caption: {generated_text}")

        sample_dict["image_id"] =  img
        sample_dict["caption_g"] = generated_text
        all_set.append(sample_dict)

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(all_set, json_file, ensure_ascii=False)


# 3. 主函数
def main(pretrained_modal_pth,source_name,json_file,source_images, target_name,target_images,target_images_dir):
    # 初始化模型和处理器
    processor = Blip2Processor.from_pretrained(pretrained_modal_pth, use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained(pretrained_modal_pth,device_map="auto")


    # 修改LoRA配置
    lora_config = LoraConfig(
        r=8,  # 降低秩大小
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.2,  # 提高dropout
        bias="none",
        modules_to_save=["language_projection"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 准备数据集
    train_dataset = JsonImageTextDataset(
        json_file=json_file,
        image_dir=source_images,
        processor=processor,
        is_train=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 训练设置
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
        weight_decay=0.01  # 显式添加L2正则
    )
    # 添加梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    model.to(device)

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")



        generate_captions(
                model=model,
                processor=processor,
                image_file=target_images,
                image_dir=target_images_dir,
                output_file=f"output\generated_S({source_name})_T({target_name})_epoch_{epoch}.source_labels"
        )



    # 保存最终模型
    model.save_pretrained("blip2_lora_finetuned")
    processor.save_pretrained("blip2_lora_finetuned")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_pth', type=str, default='Salesforce/blip2-opt-2.7b', help='')
    parser.add_argument('--source_name', type=str,default='flickr25k', help='')



    parser.add_argument('--source_labels', type=str, default='../data/flickr25k/source_labels/flickr25k_train_labels.json', help='')
    parser.add_argument('--source_images_dir', type=str, default=r"../data/flickr25k/raw/images", help='')


    parser.add_argument('--target_name', type=str, default='nuswide', help='')
    parser.add_argument('--target_images', type=str,default=r"../data/nus-wide/raw/train/images.txt", help='')
    parser.add_argument('--source_images_dir', type=str, default=r"../data/nus-wide/raw/images", help='')

    config = parser.parse_args()
    main(config.pretrained_pth,config.source_name, config.source_labels,config.source_images_dir, config.target_name,config.target_images,config.target_images_dir)
