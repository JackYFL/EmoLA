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
sys.path.append('/home/liyifa11/MyCodes/EmoDQ/EmoDQ/')
from llava.model.builder import load_pretrained_model

from llava.model import *
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import confusion_matrix
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
from rouge import Rouge
import re

emotion_list = {
    "happiness": ["happiness", "happy", "smiling",  "smile", "pleasant", "content", "contentment", "relaxed", "relaxation", "cheerful", "joy", "amusement", "amused", "positivity", "positive", "lightheartedness", "joyful", "friendly", "warmth", "bright"],
    "neutral": ["neutral", "neutrality", "calm", "contemplative", "concerned", "thoughtful", "focused", "solemn", "mild"],
    "surprise": ["surprise", "surprised", "startled", "puzzled", "astonishment", "disbelief", "wonder", "curiosity", "curious", "confusion", "wonder", "bewilderment", "skepticism", "excitement", "uncertainty"],
    "fear": ["fear", "shock", "anxiety", "apprehension", "tension", "panic", "alertness", "focus", "worry", "alarm", "curiosity", "contemplation"],
    "disgust": ["disgust", "distaste", "disgusted", "discomfort", "displeasure", "displeased", "disapproval", "disdain", "contempt"],
    "anger": ["anger", "mad", "irate", "outraged", "agitation", "irritated", "enraged", "annoyed", "incensed", "serious", "frustration", "displeased", "stern"],
    "sadness": ["sadness", "crying", "distress", "concern", "discomfort", "anguish", "somber"]
}

emotions2num = {
    "Happiness": 0,
    "Sadness": 1,
    "Surprise": 2,
    "Anger": 3,
    "Neutral": 4,
    "Disgust": 5,
    "Fear": 6
}

def remove_neg_sentences(text):
    # define list of negative words and phrases
    negative_words = ['not', 'no', "isn't ", 'nothing', 'cannot', "won't", "shouldn't", "neither", "nor"]
    # segment the text into several sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # remove sentences with negative words
    positive_sentences = [sentence for sentence in sentences if not any(negative_word in sentence.lower() for negative_word in negative_words)]
    
    return " ".join(positive_sentences)


def description2emotion(description):
    emotions = ["happiness", "sadness", "surprise", "anger", "neutral", "disgust", "fear"]
    emotions2idx = {emo:i for i, emo in enumerate(emotions)}
    emotions_cnt = {emo:0 for i, emo in enumerate(emotions)}
    
    positive_text = remove_neg_sentences(description)
    if positive_text == '':
        positive_text = description
    
    first_sentence = re.match(r'^(.*?)[.!?]', positive_text).group(1)
    remaining_text = re.sub(r'^(.*?)[.!?]', '', positive_text)
    
    first_sentence_words = re.findall(r'\b\w+\b', first_sentence.lower()) 
    for word in first_sentence_words:
        for emo in emotions:
            if word in emotion_list[emo]:
                return emotions2idx[emo]
                
    remaining_words = re.findall(r'\b\w+\b', remaining_text.lower()) 
    for word in remaining_words:
        for emo in emotions:
            if word in emotion_list[emo]:
                emotions_cnt[emo] += 1
    max_key = max(emotions_cnt, key=lambda k: emotions_cnt[k])
    
    return emotions2idx[max_key]


def parse_anno(ann_file):
    with open(ann_file, 'r') as f:
        file = f.read()
    return json.loads(file)


def cal_rouge(reference, prediction):
    rouger = Rouge()
    scores = rouger.get_scores(prediction, reference, avg=True)
    rouge_l = scores['rouge-l']
    rouge_l_f1 = rouge_l['f']
    return rouge_l_f1


def cal_bleu(reference, prediction):
    reference_word_list = [reference.split()]
    prediction_word_list = prediction.split()
    bleu_score = sentence_bleu(reference_word_list, prediction_word_list)
    
    return bleu_score


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
    elif 'gpt' in args.ann_file:
        AUs = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 20, 23, 24, 25, 26, 27, 43]
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


def AUint2Labels(AU_int_string, AU2indices):
    AUs = AU_int_string.split(', ')
    AU_labels = [0] * len(AU2indices)
    for AU in AUs:
        try:
            idx = AU2indices[AU]
            AU_labels[idx] = 1
        except:
            continue
    return AU_labels
    pass
    

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
    parser.add_argument("--task-type", type=str, default="AU")
    parser.add_argument("--model-path", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-AUgpt-only-tune-landmark-projector-gptanno_new")
    parser.add_argument("--extra-name", type=str, default=None)

    parser.add_argument("--pretrain_face_feature_projector", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-AUgpt-only-tune-landmark-projector-gptanno_new')
    parser.add_argument("--pretrain_landmark_feature_projector", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/saved_checkpoints/llava-v1.5-7b-AUgpt-only-tune-landmark-projector-gptanno_new')

    parser.add_argument("--model-base", type=str, default="/home/liyifa11/MyCodes/EmoDQ/EmoDQ/checkpoints/llava-v1.5-7b")
    parser.add_argument("--ann-file", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmotionInstructData/gpt4_annotated/AU_test.json')
    parser.add_argument("--data-dir", type=str, default='/home/liyifa11/MyCodes/EmoDQ/EmoDQ/dataset/emotion_test_dataset/test_data')
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
    if args.model_base=='None': args.model_base=None
    
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.pretrain_face_feature_projector, args.pretrain_landmark_feature_projector)

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
    annos = parse_anno(ann_file)

    if args.extra_name:
        output_dir = f'{args.output_dir}/{model_name}-{args.extra_name}/{data_name}'
    else:
        output_dir = f'{args.output_dir}/{model_name}/{data_name}'
        
    output_jsons_dir = f'{output_dir}/jsons'
    output_failed_jsons_dir = f"{output_dir}/failed_jsons"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_jsons_dir):
        os.makedirs(output_jsons_dir)
    if not os.path.exists(output_failed_jsons_dir):
        os.makedirs(output_failed_jsons_dir)
    
    # Eval
    save_dict = {}
    gt_save_dict = {}

    all_outputs, all_targets = [], []
    all_rouges = []
    all_failed_cases = []
    for img_id, img in tqdm.tqdm(enumerate(annos), desc="Processing images", unit="img_id"):
        # if not os.path.exists(f'{output_jsons_dir}/{img_id}.json'):    
            try: 
                img_path, gpt_output, img_label = img['image'], img['gpt4'], img['label'] 
                img_path_list = img_path.split('_')      
                img_path = os.path.join(data_dir, '/'.join(img_path_list[-3:]))
            except:
                img_path, img_label = img['image'], img['label']   
                new_data_dir = '/'.join(data_dir.split('/')[:-1])  
                img_path = os.path.join(new_data_dir, img_path)        
                
            query = message
            emotion_description = eval_model(args, query, img_path, model_name, model, image_processor, tokenizer)
            
            if args.task_type == 'AU':
                pred = AU2Labels(emotion_description, AU2indices)
                target = AUint2Labels(img_label, AU2indices)
                
            elif args.task_type == 'emotion':
                pred = description2emotion(emotion_description)
                try:
                    target = int(img_label)
                except:
                    target = emotions2num[img_label]
            try:
                f1 = cal_rouge(gpt_output, emotion_description)
                all_rouges.append(f1)
            except:
                pass
            
            all_outputs.append(pred)
            all_targets.append(target)
            
            if pred!=target:
                failed_case = {"image_path": img_path, "given_label": img_label, "prediction": emotion_description}
                all_failed_cases.append(failed_case)
                with open(f'{output_failed_jsons_dir}/{img_id}.json', 'w', encoding='utf-8') as f:
                    json.dump(failed_case, f, ensure_ascii=False, indent=2)
                
            save_dict[img_id] = emotion_description
            gt_save_dict[img_id] = target
            output = {"image_path": img_path, "label": emotion_description, "gt": img['label']}
            with open(f'{output_jsons_dir}/{img_id}.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
                
    with open(f'{output_dir}/fail.json', 'w', encoding='utf-8') as f:
        json.dump(all_failed_cases, f, ensure_ascii=False)
   
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    all_rouges = np.array(all_rouges)
    
    if args.task_type == 'emotion':
        acc = (all_outputs==all_targets).sum()/all_targets.shape[0]
        try:
            rouge_avg = all_rouges.mean()
            print(f"The acc is: {acc*100:.2f}%, the average rouge is: {rouge_avg*100:.2f}.")
        except:
            print("The acc is: {acc*100:.2f}%.")
            
    elif args.task_type == 'AU':
        if 'GFT' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '10', '12', '14', '15', '23', '24']
        elif 'EmotioNet' in args.ann_file:
            select_AUs = ['1', '2', '4', '5', '6', '9', '12', '17', '20', '25', '26', '43']
        elif 'DISFA' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '9', '12', '25', '26']
        elif 'BP4D' in args.ann_file:
            select_AUs = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']
        elif 'gpt' in args.ann_file:
            select_AUs = ['1', '2', '4', '5', '6', '10', '12', '17', '24', '25', '26', '43']
            
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
        rouge_avg = all_rouges.mean()
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
            try:
                result_string = f"Dataset: {data_name}, Acc: {acc*100:.2f}, Average rouge: {rouge_avg*100:.2f}%"
            except:
                result_string = f"Dataset: {data_name}, Acc: {acc*100:.2f}"
                
        elif args.task_type == 'AU':
            result_string_pre = f'Dataset: {data_name}\n'
            for AU, F1 in zip(select_AUs, F1s):
                result_string_pre += f'AU{AU}: {F1*100:.2f}, '
            result_string = result_string_pre + f"\nF1: {F1s_mean*100:.2f}%, Acc: {accs_mean*100:.2f}, Average rouge: {rouge_avg*100:.2f}%"
        print(result_string, file=f)
        print('*'*80, file = f)


if __name__ == '__main__':
    eval()