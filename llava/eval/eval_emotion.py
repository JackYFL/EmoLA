import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from llava.eval.run_llava import eval_model
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava import conversation as conversation_lib
from llava.model.builder import load_pretrained_model
from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import confusion_matrix

import json
import math
import tqdm
import argparse
import requests
import numpy as np
from typing import Dict, Optional, Sequence, List
from PIL import Image
from io import BytesIO
import re

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


def AUIndicesMap(args):
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    if 'EmotioNet' in args.ann_file:
        AUs = [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]
    elif 'DISFA' in args.ann_file:
        AUs = [1, 2, 4, 6, 9, 12, 25, 26]
    elif 'BP4D' in args.ann_file:
        AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
    elif 'GFT' in args.ann_file:
        AUs = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18, 20, 23, 24, 25, 26]
    # AU2indices = {f"AU{str(AU)}": idx for idx, AU in enumerate(AUs)}
    AU2indices = {str(AU): idx for idx, AU in enumerate(AUs)}
    return AU2indices


def AU2Labels(AU_string, AU2indices):
    pattern = r"AU(\d+)"
    matches = re.findall(pattern, AU_string)
    AU_labels = [0] * len(AU2indices)

    for num in matches:
        try:
            idx = AU2indices[num]
            AU_labels[idx] = 1
        except:
            continue
    return AU_labels


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
    # parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/LLaVA/checkpoints/llava-llava-v1.5-7b-saveall_only_query_pascolvoc")
    
    parser.add_argument("--version", type=str, default="v1")
    # parser.add_argument("--base-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/llava-v1.5-7b")
    parser.add_argument("--base-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/vicuna-7b-v1.5")
    # parser.add_argument("--task-type", type=str, default="AU")
    parser.add_argument("--task-type", type=str, default="emotion")
    parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-lora-aucls-AU_instruct_fold23_no_dup")
    # parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-vicuna-7b-v1.5-lora-after_ExpW_wo_AffectNet_wotraining_projector")
    parser.add_argument("--pretrain_query", type=bool, default=False)
    parser.add_argument("--pretrain_query_decoder", type=bool, default=False)
    parser.add_argument("--pretrain_deepquery", type=bool, default=True)

    parser.add_argument("--model-base", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/llava-v1.5-7b")
    # parser.add_argument("--model-base", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/vicuna-7b-v1.5")
    # parser.add_argument("--subdir", type=str, default='val2017')
    parser.add_argument("--ann-file", type=str, default='/egr/research-actionlab/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emotion_classification_dataset/emotion_test_dataset/annotations/RAF-DB.json')
    # parser.add_argument("--ann-file", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/AU_recognition_dataset/AU_test_dataset/annotations/DISFA/fold1.json')
    # parser.add_argument("--ann-file", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/AU_recognition_dataset/AU_test_dataset/annotations/GFT/GFT.json')
    parser.add_argument("--data-dir", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emotion_test_dataset')
    parser.add_argument("--output-dir", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/output_test')
    parser.add_argument("--num_queries", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=0.5)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--conv-mode", type=str, default=None)
    return parser.parse_args()


def eval_model(args, query, image_file, model_name, model, image_processor, tokenizer):
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
    H, W = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

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
            stopping_criteria=[stopping_criteria])

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


def get_acc_F1(confusion_matrix):
    TP = confusion_matrix[1, 1]
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]

    acc = (TP + TN) / (TP + TN + FP + FN)
    if TP == 0:
        P = 0
        R = 0
        F1 = 0
    else:
        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2*P*R/(P+R)
    return acc, F1


def eval():
    args = get_args()
    if args.task_type == 'AU':
        AU2indices = AUIndicesMap(args)
        
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    
    if 'query' in model_name:
        tokenizer, model, image_processor = initialize_model(args)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

    # import ipdb; ipdb.set_trace()
    # Form message    
    if args.task_type == 'AU':
        message = f"What AUs are conveyed by this face?"
    elif args.task_type == 'emotion':
        message = f"What emotions are conveyed by this face?"

    # Load dataset
    data_dir = args.data_dir
    ann_file = args.ann_file
    if args.task_type == 'emotion':
        data_name = ann_file.split('/')[-1].replace('.json', '')
    elif args.task_type == 'AU':
        data_name = '_'.join(ann_file.split('/')[-2:]).replace('.json', '')
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
        # if not os.path.exists(f'{output_jsons_dir}/{img_id}.json'):            
            img_path, img_label = img['image'], img['label']
            # img_path, img_label = img.keys(), img.values()
            # img_path, img_label = list(img_path)[0], list(img_label)[0]
            img_path = os.path.join(data_dir, img_path)
            query = message
            # import ipdb; ipdb.set_trace()    
            emotion = eval_model(args, query, img_path, model_name, model, image_processor, tokenizer)
            
            if args.task_type == 'AU':
                pred = AU2Labels(emotion, AU2indices)
                target = AU2Labels(img_label, AU2indices)
                
            elif args.task_type == 'emotion':
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
    # import ipdb; ipdb.set_trace()
    
    if args.task_type == 'emotion':
        acc = (all_outputs==all_targets).sum()/all_targets.shape[0]
        print(f"The acc is: {acc*100:.2f}%")
    elif args.task_type == 'AU':
        if 'GFT' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '10', '12', '14', '15', '23', '24']
        elif 'EmotioNet' in args.ann_file:
            select_AUs = ['1', '2', '4', '5', '6', '9', '12', '17', '20', '25', '26', '43']
        elif 'DISFA' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '9', '12', '25', '26']
        elif 'BP4D' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']

        F1s = []
        accs = []
        for AU in select_AUs:
            AU_id = AU2indices[AU]
            try:
                cm_au = confusion_matrix(all_targets[:, AU_id], all_outputs[:, AU_id])
                acc, F1 = get_acc_F1(cm_au)
                F1s.append(F1)
                accs.append(acc)        
            except:
                pass
        F1s, accs = np.array(F1s), np.array(accs)
        F1s_mean = np.mean(F1s)
        accs_mean = np.mean(accs)

    with open(f'{output_dir}/pred.json', 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, ensure_ascii=False)


    with open(f'{output_dir}/gt.json', 'w', encoding='utf-8') as f:
        json.dump(gt_save_dict, f, ensure_ascii=False)


    with open(f'{output_dir}/result_output.txt', 'w') as f:
        print('*'*80, file = f)
        if args.task_type == 'emotion':
            result_string = f"Dataset: {data_name}, Acc: {acc*100:.2f}%"
        elif args.task_type == 'AU':
            result_string_pre = f'Dataset: {data_name}\n'
            for AU, F1 in zip(select_AUs, F1s):
                result_string_pre += f'AU{AU}: {F1*100:.2f} '
            result_string = result_string_pre + f"\nF1: {F1s_mean*100:.2f}%, Acc: {accs_mean*100:.2f}"
        print(result_string, file=f)
        print('*'*80, file = f)


if __name__ == '__main__':
    eval()
    
    # args = get_args()

    # if args.task_type == 'AU':
    #     AU2indices = AUIndicesMap(args)

    # disable_torch_init()
    # model_name = get_model_name_from_path(args.model_path)
    
    # # if 'query' in model_name:
    # #     tokenizer, model, image_processor = initialize_model(args)
    # # else:
    # #     tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
        
    # if args.task_type == 'AU':
    #     message = f"What AUs are conveyed by this face?"
    # elif args.task_type == 'emotoin':
    #     message = f"What emotions are conveyed by this face?"
        
    # # Load dataset
    # data_dir = args.data_dir
    # ann_file = args.ann_file
    # if args.task_type == 'emotion':
    #     data_name = ann_file.split('/')[-1].replace('.json', '')
    # elif args.task_type == 'AU':
    #     data_name = '_'.join(ann_file.split('/')[-2:]).replace('.json', '')
    # # ann_file = '/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emo_instruction/emotion_instruct_wo_AffectNet.json'
    # annos = parse_anno(ann_file)

    # output_dir = f'{args.output_dir}/{model_name}/{data_name}'
    # output_jsons_dir = f'{output_dir}/jsons'
    
    # annos = os.listdir(output_jsons_dir)
    # all_outputs, all_targets = [], []
    # save_dict, gt_save_dict = {}, {}
    
    # for img_id, anno in enumerate(annos):
    #     with open(os.path.join(output_jsons_dir, anno), 'r') as file:
    #         anno_ = json.load(file)
    #     pred_label, gt_label = anno_['label'], anno_['gt']
    #     pred = AU2Labels(pred_label, AU2indices)
    #     gt = AU2Labels(gt_label, AU2indices)
    #     all_outputs.append(pred)
    #     all_targets.append(gt)
    #     save_dict[img_id] = pred
    #     gt_save_dict[img_id] = gt
        
    # all_outputs = np.array(all_outputs)
    # all_targets = np.array(all_targets)
    # # import ipdb; ipdb.set_trace()
    
    # if args.task_type == 'emotion':
    #     acc = (all_outputs==all_targets).sum()/all_targets.shape[0]
    #     print(f"The acc is: {acc*100:.2f}%")
    # elif args.task_type == 'AU':
    #     if 'GFT' in args.ann_file:
    #         select_AUs = ['1', '2', '4', '6', '10', '12', '14', '15', '23', '24']
    #     elif 'EmotioNet' in args.ann_file:
    #         select_AUs = ['1', '2', '4', '5', '6', '9', '12', '17', '20', '25', '26', '43']
    #     elif 'DISFA' in args.ann_file:
    #         select_AUs = ['1', '2', '4', '6', '9', '12', '25', '26']
    #     elif 'BP4D' in args.ann_file:
    #         select_AUs = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']

    #     F1s = []
    #     accs = []
    #     for AU in select_AUs:
    #         AU_id = AU2indices[AU]
    #         try:
    #             cm_au = confusion_matrix(all_targets[:, AU_id], all_outputs[:, AU_id])
    #             acc, F1 = get_acc_F1(cm_au)
    #             F1s.append(F1)
    #             accs.append(acc)        
    #         except:
    #             pass
    #     F1s, accs = np.array(F1s), np.array(accs)
    #     F1s_mean = np.mean(F1s)
    #     accs_mean = np.mean(accs)
        
    # with open(f'{output_dir}/pred.json', 'w', encoding='utf-8') as f:
    #     json.dump(save_dict, f, ensure_ascii=False)

    # with open(f'{output_dir}/gt.json', 'w', encoding='utf-8') as f:
    #     json.dump(gt_save_dict, f, ensure_ascii=False)

    # with open(f'{output_dir}/result_output.txt', 'w') as f:
    #     print('*'*80, file = f)
    #     if args.task_type == 'emotion':
    #         result_string = f"Dataset: {data_name}, Acc: {acc*100:.2f}%"
    #     elif args.task_type == 'AU':
    #         result_string_pre = f'Dataset: {data_name}\n'
    #         for AU, F1 in zip(select_AUs, F1s):
    #             result_string_pre += f'AU{AU}: {F1*100:.2f} '
    #         result_string = result_string_pre + f"\nF1: {F1s_mean*100:.2f}%, Acc: {accs_mean*100:.2f}"
    #     print(result_string, file=f)
    #     print('*'*80, file = f)
