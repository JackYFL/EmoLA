import os
import re
import numpy as np

output_dir = "/egr/research-actionlab/liyifa11/MyCodes/EmoDQ/EmoDQ/output_test/llava-v1.5-7b-lora-aucls-AU_instruct_fold{fold}_no_dup"
folds = ['1', '2', '3']
dataset = 'DISFA' # BP4D, DISFA, 

if 'EmotioNet' == dataset:
    AUs = ['1', '2', '4', '5', '6', '9', '12', '17', '20', '25', '26', '43']
elif 'DISFA' == dataset:
    AUs = ['1', '2', '4', '6', '9', '12', '25', '26']
elif 'BP4D' == dataset:
    AUs = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']

aus_folds = []
for fold_id in range(3):
    fold = folds[fold_id]
    dummy_folds = folds.copy()
    dummy_folds.pop(fold_id)
    output_dir_fold = output_dir.format(fold=''.join(dummy_folds))
    subdir = f'{dataset}_fold{fold}'
    fold_result = os.path.join(output_dir_fold, subdir, 'result_output.txt')
    
    if os.path.exists(fold_result):
        with open(fold_result, 'r') as file:
            content = file.read()
        matches = re.findall(r'AU\d+:\s*(\d+\.\d+)', content)
    else:
        pass
    AUs = [float(au) for au in matches]
    aus_folds.append(AUs)

aus_folds = np.array(aus_folds)
AUs_F1_mean = aus_folds.mean(axis=0)
F1_mean = aus_folds.mean(axis=1).mean()
print(AUs_F1_mean)
print(F1_mean)