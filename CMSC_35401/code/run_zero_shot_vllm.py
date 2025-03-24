# cd /net/projects/chacha/yuyang/evaluator
# conda activate /net/projects/chai-lab/miniconda3/envs/yy_notebooks
# export PATH=/net/projects/chai-lab/miniconda3/bin:$PATH # ensure using correct conda; enter after activating environment (TO-DO: check why conda fails)
# python run_zero_shot_full.py
import os
import json
import re
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams


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


    def run_label_extraction(self, tokenizer, llm, sampling_params, ls_id, df_gt_repo):
        all_prompt = []
        prompt_s = self.get_prompt(prompt_type='sys') # system prompt
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


    def run_format_check(self, tokenizer, llm, sampling_params, input_text):
        format_prompt = []
        prompt_s = self.get_prompt(check_format=True)
        format_prompt.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompt_s},
                    {"role": "user", "content": input_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        ) 

        cleaned_text = llm.generate(format_prompt, sampling_params)   
        return cleaned_text
        

    def run(self):
        # Step 1: Load Files
        df_gt_repo = pd.read_csv(self.in_csv)
        df_gt_repo['study_id'] = df_gt_repo['study_id'].apply(lambda x: str(x))
        df_gt_repo = df_gt_repo.sort_values(by='study_id').reset_index(drop=True) ###
        ls_id = list(np.unique(df_gt_repo['study_id']))
        task1_results = {}

        # Step 2: Init model
        sampling_params, tokenizer, llm = self.prepare_llm()

        # Step 3: Label Extraction
        ls_all_outputs = self.run_label_extraction(tokenizer, llm, sampling_params, ls_id, df_gt_repo)

        assert len(ls_all_outputs) == len(ls_id)
        print('Finish label extraction.')
        print('Ready to output...')


        # Step 4: Unify format
        for i in range(len(ls_id)):
            id = ls_id[i]

            generated_text = ls_all_outputs[i].outputs[0].text
            task1_match = re.search(r'<TASK1>(.*?)</TASK1>', generated_text, re.DOTALL)
            
            if task1_match:
                task1_content = task1_match.group(1).strip()
                try:
                    task1_results[id] = json.loads(task1_content)
                except:
                    print('Correcting format...')
                    cleaned_text = self.run_format_check(tokenizer, llm, sampling_params, task1_content)
                    clean_match = re.search(r'<ANSWER>(.*?)</ANSWER>', cleaned_text[0].outputs[0].text, re.DOTALL)
                    if clean_match:
                        clean_content = clean_match.group(1).strip()
                        task1_results[id] = json.loads(clean_content)
            else:
                task1_results[id] = generated_text
        
        # Step 6: Save files
        with open(self.out_dir+f'gt_results_{self.prompt_label}_cleaned.json', 'w', encoding='utf-8') as task1_file: ###
            json.dump(task1_results, task1_file, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    prompt_dir = '/net/projects/chacha/yuyang/evaluator/prompt/labeling/'
    model_pth = '/net/scratch2/yuyang/label-benchmark/LLaMA-Factory/saves/llama3-1-8b/full/sft'
    data_dir = '/net/projects/chacha/yuyang/label_benchmark/finetuning/data/'
    output_dir = '/net/projects/chacha/yuyang/evaluator/results_label/llama-small/'

    input_csv = data_dir+'eval_repo_0110.csv'

    eval_lexical = Evaluator(input_csv=input_csv,
                            output_dir=output_dir, 
                            prompt_dir=prompt_dir, 
                            model_pth = model_pth,
                            prompt='3')
    eval_lexical.run()
