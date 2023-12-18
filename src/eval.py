import json
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from configs import get_configs
from tqdm import tqdm
import torch
import tiktoken
import click
import json
import csv
import openai

def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode


def generate_gpt2(model, prompt, device):
    model.eval()
    model.to(device)
    max_new_tokens = 100
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    y = model.generate(x,
                       max_new_tokens,
                       temperature=temperature,
                       top_k=top_k)

    res = decode(y[0].cpu().tolist())
    end = res.find("<|endoftext|>")
    if end > 0:
        return res[:end]
    else:
        return res


@click.command()
@click.option('--sft', '-s')
@click.option('--name', '-n') #AdamW
def main(sft,name):

    with open("prompts.csv", encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        prompts = [row["prompt"] for row in reader]

    print("Run inference")
    if os.path.exists("responses_"+name+".json"):
        with open("responses_"+name+".json") as fp:
            responses = json.load(fp)
    else:
        device = "cuda"
        cfg = get_configs("gpt2-medium")
        with torch.inference_mode():
            gpt_vanilla = torch.compile(GPT.from_pretrained(cfg))
            gpt_sft = torch.compile(GPT.from_checkpoint(
                cfg,
                sft))

            responses = []
            for prompt in tqdm(prompts):
                responses.append({
                    "vanilla": generate_gpt2(gpt_vanilla, f"Human: {prompt}\n\nAssistant: ", device)[
                               len(f"Human: {prompt}\n\nAssistant: "):],
                    "sft": generate_gpt2(gpt_sft, f"Human: {prompt}\n\nAssistant: ", device)[
                           len(f"Human: {prompt}\n\nAssistant: "):],
                    "prompt": prompt
                })
            with open("responses_"+name+".json", "w") as fp:
                json.dump(responses, fp)

    # Initialize reward model
    reward_name = "/mntcephfs/lab_data/mazhuoheng/MDS5210-23fall/src/reward-model-deberta-v3-large-v2"
    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

    
    print("Query reward model")
    favor_sft_over_vanilla, favor_vanilla_over_sft = 0, 0
    sft_vanilla = []
    for response in tqdm(responses):
        question, answer_a, answer_b = prompt=response["prompt"], response["vanilla"], response["sft"]
        
        input_a = tokenizer(question, answer_a, return_tensors='pt')
        input_b = tokenizer(question, answer_b, return_tensors='pt')
        score_a = rank_model(**input_a).logits[0].cpu().detach().item()
        score_b = rank_model(**input_b).logits[0].cpu().detach().item()

        if score_a > score_b:
            favor_vanilla_over_sft += 1
            sft_vanilla.append({
                "winner": "vanilla",
                "sft": response["sft"],
                "vanilla": response["vanilla"],
            })
        elif score_a <= score_b:
            favor_sft_over_vanilla += 1
            sft_vanilla.append({
                "winner": "sft",
                "sft": response["sft"],
                "vanilla": response["vanilla"],
            })
        else:
            pass
    print("favor_sft_over_vanilla", favor_sft_over_vanilla,
          favor_sft_over_vanilla / (favor_vanilla_over_sft + favor_sft_over_vanilla))
    print("favor_vanilla_over_sft", favor_vanilla_over_sft,
          favor_vanilla_over_sft / (favor_vanilla_over_sft + favor_sft_over_vanilla))

    with open("reward_preferences_"+name+".json", "w") as fp:
        json.dump(sft_vanilla, fp)




if __name__ == '__main__':
    main()
