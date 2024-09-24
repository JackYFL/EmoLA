import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field

from llava.eval.run_llava import eval_model
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava import conversation as conversation_lib

import sys
sys.path.append('~/FABA/EmoLA/')
from emollava.model.builder import load_pretrained_model

from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig

import json
import math
import tqdm
import ast
import argparse
import requests
import numpy as np
from typing import Dict, Optional, Sequence, List
from PIL import Image
from io import BytesIO

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def parse_anno(ann_file):
    with open(ann_file, 'r') as f:
        file = f.read()
    return json.loads(file)


def emotion2target(emotion):
    emotions_map = {"Surprise": 0,
                    "Fear": 1,
                    "Disgust": 2,
                    "Happiness": 3,
                    "Sadness": 4,
                    "Anger": 5,
                    "Neutral": 6}
    if emotion in emotions_map:
        return emotions_map[emotion]
    else:
        return 7

        
def initialize_model(training_args, load_8bit=False, load_4bit=False, device_map="auto"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
        
    model = LlavaLlamaForCausalLMDeepQuery.from_pretrained(
        training_args.base_path,
        low_cpu_mem_usage=True, 
        **kwargs
    )

    training_args.pretrain_query_path = training_args.model_path + '/mm_query.bin'
    training_args.pretrain_query_decoder_path = training_args.model_path + '/mm_query_decoder.bin'
    training_args.pretrain_query_bert_path = training_args.model_path + '/mm_query_bert.bin'
    training_args.pretrain_deepquery_path = training_args.model_path + '/deep_query.bin'
    training_args.pretrain_icondeepquery_path = training_args.model_path + '/icondeepquery.bin'

    if training_args.pretrain_query:
        model.get_model().initialize_queries(training_args)
    if training_args.pretrain_query_decoder:
        model.get_model().initialize_query_decoder(training_args)
    if training_args.pretrain_query_bert:
        model.get_model().initialize_query_bert(training_args)
    if training_args.pretrain_deepquery:
        model.initialize_deepquery(training_args)
    if training_args.pretrain_icondeepquery:
        model.initialize_icondeepquery(training_args)
        model.config.pretrain_icondeepquery = True
        
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(training_args.base_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    
    if training_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[training_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    return tokenizer, model, image_processor


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--base-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/llava-v1.5-7b")
    parser.add_argument("--data_name", type=str, default="MMAFEDB")
    # parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-lora-emocls_face_landmark_stage2")
    # parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-lora-emocls_nostage1-onlyfacefeatures_stage2")
    parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-lora-aucls-AU_instruct_fold23_no_dup")
    parser.add_argument("--pretrain_query", type=bool, default=False)
    parser.add_argument("--pretrain_query_decoder", type=bool, default=False)
    parser.add_argument("--pretrain_query_bert", type=bool, default=False)
    parser.add_argument("--pretrain_deepquery", type=bool, default=False)
    parser.add_argument("--pretrain_icondeepquery", type=bool, default=False)
    parser.add_argument("--pretrain_face_feature_projector", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-emocls_face_landmark')
    parser.add_argument("--pretrain_landmark_feature_projector", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-emocls_face_landmark')

    parser.add_argument("--model-base", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/llava-v1.5-7b")
    parser.add_argument("--ann-file", type=str, default='/egr/research-actionlab/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emotion_classification_dataset/emotion_test_dataset/annotations/RAF-DB.json')
    parser.add_argument("--data-dir", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emotion_test_dataset')
    parser.add_argument("--output-dir", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/output_test')
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--conv-mode", type=str, default=None)
    return parser.parse_args()


def eval_model(args, query, image_file, face_feature_path, landmark_feature_path, model_name, model, image_processor, tokenizer):
    qs = query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = load_image(image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    org_face_feature, org_landmark_feature = np.load(face_feature_path), np.load(landmark_feature_path)
    org_face_feature, org_landmark_feature = torch.tensor(org_face_feature).half().cuda(), torch.tensor(org_landmark_feature).half().cuda()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # import ipdb; ipdb.set_trace()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            top_p=args.top_p,
            # num_beams=args.num_beams,
            temperature=0.01,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            org_face_features=org_face_feature, 
            org_landmark_features=org_landmark_feature)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


def eval():
    args = get_args()
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    
    if 'query' in model_name:
        tokenizer, model, image_processor = initialize_model(args)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.pretrain_face_feature_projector, args.pretrain_landmark_feature_projector)

    # import ipdb; ipdb.set_trace()
    # Form message    
    message = f"What emotions are conveyed by this face?"

    # Load dataset
    data_dir = args.data_dir
    ann_file = args.ann_file
    data_name = ann_file.split('/')[-1].replace('.json', '')
    # ann_file = '/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emo_instruction/emotion_instruct_wo_AffectNet.json'
    annos = parse_anno(ann_file)

    output_dir = f'{args.output_dir}/{model_name}/{data_name}'
    output_jsons_dir = f'{output_dir}/jsons'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_jsons_dir):
        os.makedirs(output_jsons_dir)
    
    # Eval
    save_dict = {}
    gt_save_dict = {}

    all_outputs, all_targets = [], []
    for img_id, img in tqdm.tqdm(enumerate(annos), desc="Processing images", unit="img_id"):
        if not os.path.exists(f'{output_jsons_dir}/{img_id}.json'):            
            img_path, img_label = img['image'], img['label']
            img_face_feature, img_landmark_feature = img['face_embedding'], img['landmark_embedding']
            img_face_feature_path = os.path.join(data_dir, img_face_feature)
            img_landmark_feature_path = os.path.join(data_dir, img_landmark_feature)
            img_path = os.path.join(data_dir, img_path)
            query = message
            
            emotion = eval_model(args, query, img_path, img_face_feature_path, img_landmark_feature_path, model_name, model, image_processor, tokenizer)
            pred = emotion2target(emotion)
            target = emotion2target(img_label)

            all_outputs.append(pred)
            all_targets.append(target)

            save_dict[img_id] = emotion
            gt_save_dict[img_id] = target
            output = {"image_path": img_path, "label": emotion, "gt": img['label']}
            with open(f'{output_jsons_dir}/{img_id}.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False)
                
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    acc = (all_outputs==all_targets).sum()/all_targets.shape[0]
    print(f"The acc is: {acc*100:.2f}%")

    with open(f'{output_dir}/pred.json', 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False)

    with open(f'{output_dir}/gt.json', 'w', encoding='utf-8') as f:
        json.dump(gt_save_dict, f, ensure_ascii=False)

    with open(f'{output_dir}/result_output.txt', 'w') as f:
        print('*'*80, file = f)
        result_string = f"Dataset: {data_name}, Acc: {acc*100:.2f}%"
        print(result_string, file=f)
        print('*'*80, file = f)


if __name__ == '__main__':
    eval()