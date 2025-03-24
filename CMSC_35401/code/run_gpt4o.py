# cd /data/yuyang/gpt_4v
# conda activate /data/anaconda3/envs/cxr-eval
# python test_gpt4o.py --type 8 --p /data/yuyang/gpt_4v/gpt_prompts/ --i /data/yuyang/gpt_4v/human_study/ --o /data/yuyang/gpt_4v/results_4o/
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re
from openai import AzureOpenAI
import json

CXR_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", 
"Lung Opacity", "Edema", "Consolidation", "Pneumonia", 
"Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]


def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, dest='api_key', 
                        help="API key for Azure API.")
    parser.add_argument("--endpoint", type=str, dest='endpoint', 
                        help="Endpoint for Azure API.")
    parser.add_argument("--i", type=str, 
                        dest="input_dir", default=None, 
                        help="Directory path to input csv(s).")
    parser.add_argument("--o", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output csv(s).")
    parser.add_argument("--p", type=str, 
                        dest="prompt_dir", default=None, 
                        help="Directory path to prompts.")
    parser.add_argument("--type", type=str, default='1', 
                        help="Choose the type of prompts you want to input.")
    args = parser.parse_known_args()

    return args


# (In) gt_indc_sample.csv + ind_list_sample.csv 
# (Out) gen_findings_sample.csv + gen_imp_sample.csv + err_list.txt : study_id (str), report (str)
class GenLabelAzureGPT:
  def __init__(self, api_key, endpoint, input_dir, output_dir, prompt_dir, max_retries=3, prompt='1', labels=CXR_LABELS):
      self.prompt_label = prompt
      self.api_key = api_key
      self.endpoint = endpoint
      self.in_repo_pth = input_dir + '20_gt_repo.csv' ###
      self.out_dir = output_dir
      self.max_retries = max_retries
      self.prompt_dir = prompt_dir



  def gen_per_request(self, report, api_version='2024-02-01', deployment_name='gpt-4o-test'): ### api_version: 2023-12-01-preview, deployment_name=gpt-4v
      '''
      Report Generation for each request.
      '''

      client = AzureOpenAI(
          api_key=self.api_key,  
          api_version=api_version,
          base_url=f"{self.endpoint}/openai/deployments/{deployment_name}"
      )

      # user text prompt
      with open(os.path.join(self.prompt_dir, f'eval_prompt_{self.prompt_label}.txt'), 'r') as file:
          prompt = file.read()

      # text+image prompt
      apiresponse = client.chat.completions.with_raw_response.create(
          model=deployment_name,
          messages=[
              {
                  "role": "system", 
                  "content": prompt
                  },
              {
                  "role": "user",
                  "content": report
                  }
          ],
          max_tokens=4096
      )

      debug_sent = apiresponse.http_request.content
      chat_completion = apiresponse.parse()
      response = chat_completion.choices[0].message.content
      
      return response

  

  def run(self):
    # Step 1: Load Files
    df_gt_repo = pd.read_csv(self.in_repo_pth).sort_values(by='study_id').reset_index(drop=True)
    ls_id = list(np.unique(df_gt_repo['study_id']))
    task1_results = {}
    alert_results = {}

    # Step 2: LLM Generation
    for i in tqdm(range(len(ls_id))):
        id = ls_id[i]
        repo = list(df_gt_repo[df_gt_repo['study_id']==id]['report'])[0]
        generated_text = self.gen_per_request(repo)

        task1_match = re.search(r'<TASK1>(.*?)</TASK1>', generated_text, re.DOTALL)
        alert_match = re.search(r'<ALERT>(.*?)</ALERT>', generated_text, re.DOTALL)
        
        if task1_match:
            task1_content = task1_match.group(1).strip()
            try:
                task1_results[id] = json.loads(task1_content)
            except:
                task1_results[id] = task1_content
        else:
            task1_results[id] = generated_text

        if alert_match:
            alert_content = alert_match.group(1).strip()
            alert_results[id] = alert_content.split('\n')
        else:
            alert_results[id] = generated_text
    
    # Step 3: Save json
    with open(self.out_dir+f'task1_results_{self.prompt_label}.json', 'w', encoding='utf-8') as task1_file:
        json.dump(task1_results, task1_file, ensure_ascii=False, indent=4)

    with open(self.out_dir+f'alert_results_{self.prompt_label}.json', 'w', encoding='utf-8') as alert_file:
        json.dump(alert_results, alert_file, ensure_ascii=False, indent=4)
           

    

if __name__ == '__main__':
  args, _ = parse_args()

  GenRepo = GenLabelAzureGPT(api_key=args.api_key,
                            endpoint=args.endpoint, 
                            input_dir=args.input_dir,
                            output_dir=args.output_dir, 
                            prompt_dir=args.prompt_dir, 
                            prompt=args.type, 
                            max_retries=3)
  GenRepo.run()