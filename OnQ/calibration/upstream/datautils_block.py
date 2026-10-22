# Source lineage: yasu0001/ReasoningQAT
# Upstream revision: c36060fefb05aa8c29dcad47b52feacf95201d37
# Original file: datautils_block.py; payload preserved, provenance header added.
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset
import os

def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def format_openthoughts_sample(example):
    """Convert OpenThoughts format to HuggingFace chat format"""
    messages = []
    for item in example:
        if item["from"] == "human":
            messages.append({
                "role": "user",
                "content": item["value"]
            })
        elif item["from"] == "gpt":
            messages.append({
                "role": "assistant", 
                "content": item["value"]
            })
    return {
        "messages": messages
    }

def get_openthoughts_shuffled(tokenizer, train_size, val_size, seed, seqlen, test_only=False):
    print("get_openthoughts shuffled")
    traindata = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)

    trainloader = []
    valloader = []
    
    target_seqlen = seqlen  # Fixed sequence length for all samples
    
    for i, sample in enumerate(traindata):
        if len(trainloader) >= train_size and len(valloader) >= val_size:
            break
        
        # Format conversation using the same method as main_e2e_distill.py
        try:
            formatted_sample = format_openthoughts_sample(sample['conversations'])
            trainchat = tokenizer.apply_chat_template(formatted_sample['messages'], tokenize=False)
            trainenc = tokenizer(trainchat, return_tensors='pt')
            
            # Only use samples with at least target_seqlen tokens
            if trainenc.input_ids.shape[1] >= target_seqlen:
                # Truncate to exactly target_seqlen tokens from beginning
                inp = trainenc.input_ids[:, :target_seqlen]
                tar = inp.clone()
                
                if len(trainloader) < train_size:
                    trainloader.append((inp, tar))
                elif len(valloader) < val_size:
                    valloader.append((inp, tar))
        except Exception as e:
            # Skip samples that cause tokenization errors
            continue
    
    print(f"OpenThoughts: collected {len(trainloader)} train samples and {len(valloader)} val samples")
    return trainloader, valloader

def get_redpajama_concat(tokenizer, train_size, val_size, seed, seqlen, test_only=False):
    print("get_redpajama_concat")
    random.seed(seed)

    target_seqlen = seqlen
    localtrainloader = []
    localvalloader = []
    bos_token =  tokenizer.bos_token if tokenizer.bos_token else tokenizer.additional_special_tokens[0]
    # this is for Qwen
    eos_token = tokenizer.eos_token
    data_buffer = ""

    split_train_size = train_size
    split_val_size = val_size
    # general data
    general_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, sample in enumerate(general_dataset):
        if len(localtrainloader) >= split_train_size and len(localvalloader) >= split_val_size:
            break

        data_buffer += bos_token + sample['text'] + eos_token
        tokenized = tokenizer(data_buffer, return_tensors='pt')
        if tokenized.input_ids.shape[1] >= target_seqlen:
            inp = tokenized.input_ids[:, :target_seqlen]
            tar = inp.clone()
            if len(localtrainloader) < split_train_size:
                localtrainloader.append((inp, tar))
            elif len(localvalloader) < split_val_size:
                localvalloader.append((inp, tar))
            
            data_buffer = ""
    return localtrainloader, localvalloader

def get_sweep_dataset(tokenizer, train_size, val_size, seed, seqlen, test_only=False, reasoning_ratio=0.5):
    print("get_sweep_dataset")
    target_seqlen = seqlen
    reasoning_train_size = int(train_size * reasoning_ratio)
    reasoning_val_size = int(val_size * reasoning_ratio)
    non_reasoning_train_size = train_size - reasoning_train_size
    non_reasoning_val_size = val_size - reasoning_val_size
    
    if reasoning_ratio != 0.0: 
        reasoning_trainloader, reasoning_valloader = get_openthoughts_shuffled(tokenizer, reasoning_train_size, reasoning_val_size, seed, seqlen, test_only)
    else:
        reasoning_trainloader, reasoning_valloader = [], []

    if reasoning_ratio != 1.0:
        non_reasoning_trainloader, non_reasoning_valloader = get_redpajama_concat(tokenizer, non_reasoning_train_size, non_reasoning_val_size, seed, seqlen, test_only)
    else:
        non_reasoning_trainloader, non_reasoning_valloader = [] , []

    trainloader = reasoning_trainloader + non_reasoning_trainloader
    valloader = reasoning_valloader + non_reasoning_valloader
    random.shuffle(trainloader)
    random.shuffle(valloader)
    return trainloader, valloader

def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False, **kwargs
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'sweep_1.0' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=1.0)
    elif 'sweep_0.95' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.95)
    elif 'sweep_0.9' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.9)
    elif 'sweep_0.8' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.8)
    elif 'sweep_0.6' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.6)
    elif 'sweep_0.4' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.4)
    elif 'sweep_0.2' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.2)
    elif 'sweep_0.0' == name:
        return get_sweep_dataset(tokenizer,train_size,val_size,seed,seqlen,test_only,reasoning_ratio=0.0)
    else:
        raise NotImplementedError



@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model,'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results


class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data', off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size%batch_size == 0
         
        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            # self.data = torch.zeros((self.size//self.batch_size, self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            self.data = [None] * (self.size // self.batch_size)

    def _initialize_data_on_disk(self):
        for idx in range(self.size//self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size//self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            target = self.data[idx]
            if target is None or target.shape != new_data.shape or target.dtype != new_data.dtype:
                # 初回のみ確保
                self.data[idx] = torch.empty_like(new_data)
                target = self.data[idx]
            target.copy_(new_data)  # ← 置換ではなく上書き

