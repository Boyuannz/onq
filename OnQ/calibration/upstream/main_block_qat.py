# Source lineage: yasu0001/ReasoningQAT
# Upstream revision: c36060fefb05aa8c29dcad47b52feacf95201d37
# Original file: main_block_qat.py; local latent-checkpoint save hooks added.
import os
import sys
import random
import numpy as np
import torch
import time
from datautils_block import get_loaders, test_ppl
import torch.nn as nn
from quantize.block_qat import block_ap
from tqdm import tqdm
import utils
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model



torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        # print(f"task_list: {task_list}")
        model = HFLM(pretrained=model, tokenizer=tokenizer)
        # task_manager = TaskManager()
        # task_names = task_manager.match_tasks(task_list)
        print(f"task_names: {task_list}")
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            # task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            if task in results['results'] and 'acc,none' in results['results'][task]:
                total_acc += results['results'][task]['acc,none']
        if len(task_list) > 0:
            logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
        else:
            logger.info('No valid tasks found for accuracy calculation')
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="direction of cached dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--save_latent_dir", default=None, type=str,
                        help="separate pre-round training model; defaults beside quantized export")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--fake_quant", default=False, action="store_true",
                        help="use fake quantization instead of real quantization, can reduce memory footprint (used only for evaluation)")
    parser.add_argument("--resume_quant", type=str, default=None,  help="model path of resumed quantized model")
    parser.add_argument("--calib_dataset",type=str,default="sweep_0.8",
        choices=["sweep_0.0", "sweep_0.2", "sweep_0.4", "sweep_0.6", "sweep_0.8", "sweep_0.9", "sweep_0.95", "sweep_1.0"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--net", type=str, default=None,help="model (family) name, for the easier saving of data cache")
    parser.add_argument("--max_memory", type=str, default="16GiB",help="The maximum memory of each GPU")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--kd_loss", action="store_true", default=False, help="use knowledge distillation loss")
    parser.add_argument("--kd_loss_weight", type=float, default=0.5, help="weight of knowledge distillation loss")
    parser.add_argument("--discard_outlier", action="store_true", default=False, help="discard outlier in loss calculation")
    parser.add_argument("--outlier_loss_weight", type=float, default=1.0, help="weight of outlier loss")
    parser.add_argument("--quantizer_class", type=str, default="UniformAffineQuantizer", help="quantizer class to use (e.g., UniformAffineQuantizer, LogQuantizer)")
    parser.add_argument("--scale", type=float, default=0.7, help="scale of quantizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")

    # Outlier loss parameters
    parser.add_argument("--outlier_pct", type=float, default=0.0, help="percentile for outlier detection (0.0 to disable)")
    parser.add_argument("--ignore_first_token", action="store_true", default=False, help="ignore first token in loss calculation")
    parser.add_argument("--normalize", action="store_true", default=False, help="normalize features before loss calculation")
    
    # Input dataset selection parameters
    parser.add_argument("--use_fp_inputs", action="store_true", default=False, help="use fp_train_inps instead of quant_train_inps for training modules")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of layer reconstruction loss")
    parser.add_argument("--beta", type=float, default=0.5, help="weight of self_attn reconstruction loss")
    parser.add_argument("--nblock", type=int, default=1, help="number of blocks to quantize simultaneously (default: 1)")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    logger.info(f"check the availability of cuda: is_available: {torch.cuda.is_available()}")

    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")
    if args.resume_quant:
        # directly load quantized model for evaluation
        if args.real_quant:
            from quantize.int_linear_real import load_quantized_model
            model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.group_size)
        elif args.fake_quant:
            from quantize.int_linear_fake import load_quantized_model
            model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.group_size, replace=True, strict=True, quantizer_class=args.quantizer_class, inference_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.resume_quant, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.resume_quant, device_map='cuda', torch_dtype=torch.float16, trust_remote_code=True)
        logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        if os.path.exists(args.model):
            # Extract wbits and group_size from model path
            # Expected format: output/block_ap_log/Qwen3-0.6B-w4g128
            model_name = os.path.basename(args.model)
            wbits_from_path = None
            group_size_from_path = None
            
            # Try to extract wbits and group_size from model name
            if 'w' in model_name and 'g' in model_name:
                try:
                    # Extract the last part after splitting by '-'
                    # For "Qwen3-0.6B-w4g128", this gives "w4g128"
                    last_part = model_name.split('-')[-1]
                    if last_part.startswith('w') and 'g' in last_part:
                        wbits_str, group_size_str = last_part[1:].split('g')
                        wbits_from_path = int(wbits_str)
                        group_size_from_path = int(group_size_str)
                        logger.info(f"Extracted wbits={wbits_from_path}, group_size={group_size_from_path} from model path")
                except (ValueError, IndexError):
                    logger.warning("Could not extract wbits/group_size from model path, using default values")
            
            # Use extracted values if available, otherwise use args values
            current_wbits = wbits_from_path if wbits_from_path is not None else args.wbits
            current_group_size = group_size_from_path if group_size_from_path is not None else args.group_size

            from quantize.int_linear_fake import load_quantized_model
            model, tokenizer = load_quantized_model(args.model, current_wbits, current_group_size, replace=True, strict=True, quantizer_class=args.quantizer_class)
            logger.info(f"Loaded quantized model with wbits={current_wbits}, group_size={current_group_size}")
        else:
            # load fp quantized model
            config = AutoConfig.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, legacy=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cuda', torch_dtype=torch.float16, trust_remote_code=True, attn_implementation='flash_attention_2')
        for param in model.parameters():
            param.requires_grad = False

        # quantization
        if args.wbits < 16:
            logger.info("=== start quantization ===")
            tick = time.time()     
            # load calibration dataset
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}, len(trainloader): {len(trainloader)}")
                # valloader = torch.load(cache_valloader)
                valloader = None
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)    
                torch.save(valloader, cache_valloader)    
            block_ap(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()
    latent_writer = getattr(args, '_latent_checkpoint', None)
    if latent_writer is not None:
        latent_writer.finalize(model, tokenizer)
        logger.info(f"saved pre-round latent model: {latent_writer.output}")
    if args.save_quant_dir:
        logger.info("start saving model")
        model.save_pretrained(args.save_quant_dir)  
        tokenizer.save_pretrained(args.save_quant_dir) 
        if latent_writer is not None:
            from OnQ.quantization.latent_checkpoint import mark_materialized_export
            mark_materialized_export(args.save_quant_dir, latent_writer.output)
        logger.info("save model success")
    evaluate(model, tokenizer, args,logger)



if __name__ == "__main__":
    print(sys.argv)
    main()
