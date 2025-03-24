# cd /net/projects/chacha/yuyang/evaluator
# conda activate /net/projects/chai-lab/miniconda3/envs/yy_notebooks
# export PATH=/net/projects/chai-lab/miniconda3/bin:$PATH # ensure using correct conda; enter after activating environment (TO-DO: check why conda fails)
# python run_multi_step_full.py
import os
import json
import re
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

CXR_LABELS_1 = ['Enlarged Cardiomediastinum', 'Cardiomegaly',
       'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
       'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
       'Fracture', 'Support Devices'] ## exclude No Finding


class Evaluator:
    def __init__(self, input_csv, output_dir, model_pth, prompt_dir, prompt):
        self.prompt_label = prompt
        self.in_csv = input_csv
        self.out_dir = output_dir
        self.prompt_dir = prompt_dir
        self.model_pth = model_pth
    

    def get_prompt(self, prompt_type='', check_format=False):
        if check_format:
            with open(os.path.join(self.prompt_dir, f'sys_prompt_{self.prompt_label}_format.txt'), 'r', encoding='utf-8') as file:
                prompt = file.read()
        else:
            with open(os.path.join(self.prompt_dir, f'{prompt_type}_prompt_{self.prompt_label}.txt'), 'r', encoding='utf-8') as file:
                prompt = file.read()
        
        return prompt        


    def prepare_llm(self):
        sampling_params = SamplingParams(temperature=1e-5, max_tokens=4096)
        llm = LLM(self.model_pth, tensor_parallel_size=4) # for 4*A100
        tokenizer = llm.get_tokenizer()

        return sampling_params, tokenizer, llm


    def run_label_extraction(self, tokenizer, llm, sampling_params, ls_id, df_gt_repo, label):
        all_prompt = []
        prompt_s = self.get_prompt(prompt_type='sys').format(label) # system prompt
        for i in range(len(ls_id)):
            id = ls_id[i]
            imp = list(df_gt_repo[df_gt_repo['study_id']==id]['imp'])[0]
            findings = list(df_gt_repo[df_gt_repo['study_id']==id]['findings'])[0]
            prompt_u = self.get_prompt(prompt_type='usr').format(findings, imp) # findings+imp

            all_prompt.append(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompt_s},
                        {"role": "user", "content": prompt_u},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        
        ls_all_outputs = llm.generate(all_prompt, sampling_params)

        return ls_all_outputs
    
    def split_label(self, ls_gen_text):
        task1_results = []
        for i in range(len(ls_gen_text)):
            generated_text = ls_gen_text[i]

            task1_match = re.search(r'<ANSWER>(.*?)</ANSWER>', generated_text, re.DOTALL)
            
            if task1_match:
                task1_content = task1_match.group(1).strip()
                task1_results.append(task1_content)
            else:
                task1_results.append('')
        
        return task1_results
        

    def run(self):
        # Step 1: Load Files
        df_gt_repo = pd.read_csv(self.in_csv)
        df_gt_repo['study_id'] = df_gt_repo['study_id'].apply(lambda x: str(x))
        df_gt_repo = df_gt_repo.sort_values(by='study_id').reset_index(drop=True) ###
        ls_id = list(np.unique(df_gt_repo['study_id']))
        
        df_gen_labels = pd.DataFrame(index=range(len(ls_id)), columns=['study_id']+CXR_LABELS_1)
        df_gen_labels['study_id'] = ls_id

        # Step 2: Init model
        sampling_params, tokenizer, llm = self.prepare_llm()

        # Step 3: Label Extraction
        n=0
        for condition in CXR_LABELS_1:
            n+=1
            print(f'Ready to extract {n}/13 medical condition...')
            ls_all_outputs = self.run_label_extraction(tokenizer, llm, sampling_params, ls_id, df_gt_repo, label=condition)
            assert len(ls_all_outputs) == len(ls_id)
            ls_gen_text = [ls_all_outputs[i].outputs[0].text for i in range(len(ls_all_outputs))]

            ls_gen_labels = self.split_label(ls_gen_text)

            df_gen_labels[condition] = ls_gen_labels


        
        print('Finish label extraction.')
        print('Ready to output...')

        df_gen_labels.to_csv(self.out_dir+f'gt_results_{self.prompt_label}.csv', index=False)



if __name__ == '__main__':

    prompt_dir = '/net/projects/chacha/yuyang/evaluator/prompt/labeling/'
    model_pth = '/net/scratch/llama/Meta-Llama-3.1-70B-Instruct'
    data_dir = '/net/projects/chacha/yuyang/evaluator/human_labels/'
    output_dir = '/net/projects/chacha/yuyang/evaluator/results_label/'

    input_csv = data_dir+'curate_gt_reports_sections.csv'

    eval_lexical = Evaluator(input_csv=input_csv,
                            output_dir=output_dir, 
                            prompt_dir=prompt_dir, 
                            model_pth = model_pth,
                            prompt='2')
    eval_lexical.run()
