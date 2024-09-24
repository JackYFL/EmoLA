import os
import json
import tqdm
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from torch.utils.data import DataLoader
import onnxruntime
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
import onnx

class LandmarkDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_landmark_emb(json_file, device, source, onnx_model_path):
    root_file = f"./emotion_landmark_features/{source}"

    if not os.path.exists(root_file):
        os.makedirs(root_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    paths = [entry['image'] for entry in data]
    emb_dict = {}

    def process_image(fn):
        img = cv2.imread(fn)
        resized_img = cv2.resize(img, (192, 192))
        normalized_img = resized_img.astype(np.float32) / 255.0
        transposed_img = np.transpose(normalized_img, (2, 0, 1))
        return np.array(transposed_img), fn

    # Using multi-threading for image loading and resizing
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(tqdm.tqdm(executor.map(process_image, paths),
                                 total=len(paths), desc='Open&Resize images'))

    images_resized, paths = zip(*results)
    images_resized = np.stack(images_resized)

    # Define your dataset for landmark extraction
    dataset = LandmarkDataset(images_resized)
    loader = DataLoader(
        dataset,
        num_workers=24,
        batch_size=1,
        shuffle=False,
        drop_last=False
    )

    onnx_model = onnx.load(onnx_model_path)
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    emb = []
    output_emb = []
    for input_image in tqdm.tqdm(loader, desc='Generate Landmark Embedding', total=len(loader)):
        input_image = np.array(input_image)
        input_name = onnx_model.graph.input[0].name
        output = ort_session.run(None, {input_name: input_image})
        # Assuming this is the embedding layer
        embedding_vector = np.squeeze(np.array(output[0][0]))
        # print(embedding_vector.shape)
        emb.append(embedding_vector)

    emb = np.array(emb)
    for i, image_path in enumerate(paths):
        emb_dict[image_path] = {'landmark_emb': emb[i]}
        output_emb.append({'image': image_path, 'embedding': emb[i]})

        image_folder = image_path.split('/')[-2]
        print(image_folder)
        image_name = '.'.join(image_path.split('/')[-1].split('.')[:- 1])

        image_folder = os.path.join(root_file, image_folder)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        output_file = os.path.join(image_folder, f"{image_name}.npy")
        np.save(output_file, emb[i])
        print(output_file)

    return emb_dict


emb_dict = get_landmark_emb(
    json_file="./annotations/GFT/GFT.json",
    device="cuda",
    source="Aff-Wild2",
    onnx_model_path="./2d106det.onnx"
)
