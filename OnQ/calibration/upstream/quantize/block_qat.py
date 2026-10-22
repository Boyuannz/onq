# Source lineage: yasu0001/ReasoningQAT
# Upstream revision: c36060fefb05aa8c29dcad47b52feacf95201d37
# Original file: quantize/block_qat.py; local latent-checkpoint save hooks added.
import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import gc
import psutil
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters, set_layernorm_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
import transformers.models.llama.modeling_llama
from OnQ.quantization.latent_checkpoint import BlockwiseLatentWriter
from OnQ.calibration.parallel import make_forward

def evaluate_loss(qlayer, fp_train_inps, quant_train_inps, dev, attention_mask, position_ids, position_embeddings, block_index, loss_func):
    losses= []
    for index, (label, quant_inp) in enumerate(zip(fp_train_inps, quant_train_inps)):
        if index == 32:
            break
        with torch.no_grad():
            quant_inp = quant_inp.to(dev)
            label = label.to(dev)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Transfer position_embeddings tuple to GPU
                pos_emb_gpu = tuple(emb.to(dev) for emb in position_embeddings[index])
                output = qlayer(quant_inp, attention_mask=attention_mask, position_ids=position_ids[index].to(dev), position_embeddings=pos_emb_gpu)
                # output = output * scale_factor
            loss = loss_func(output.float(), label.float())
            losses.append(loss.mean().item())
    return sum(losses)/len(losses)


def update_dataset(layer, dataset, dev, attention_mask, position_ids, position_embeddings=None, logger=None):
    forward_layer = make_forward(layer)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                # Transfer position_embeddings tuple to GPU
                pos_emb_gpu = tuple(emb.to(dev) for emb in position_embeddings[index])
                new_data = forward_layer(inps, attention_mask=attention_mask,position_ids=position_ids[index].to(dev), position_embeddings=pos_emb_gpu).to('cpu')
                # detect whether new data is nan
                if not torch.isfinite(new_data).all():
                    raise FloatingPointError(f"Nonfinite activation at batch {index}")
                dataset.update_data(index,new_data)
                
                # Log CPU memory usage every iteration to detect memory leaks
                if logger is not None:
                    cpu_memory_mb = psutil.virtual_memory().used / 1024**2
                    cpu_memory_percent = psutil.virtual_memory().percent
                    gpu_memory_mb = torch.cuda.max_memory_allocated(dev) / 1024**2
                    logger.info(f"update_dataset iter {index}: cpu_memory:{cpu_memory_mb:.1f}MB({cpu_memory_percent:.1f}%) gpu_memory:{gpu_memory_mb:.1f}MB")
                
                del inps, new_data
    # A full gc.collect() per batch (~0.2 s) dominated Block-AP wall time; run it once per pass.
    torch.cuda.empty_cache()
    gc.collect()

from functools import partial

def make_output_data(layer, next_layer, data, dev, attention_mask, position_id, position_embedding=None):

    tmp_outputs = dict()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            input = data.to(dev)
            if len(input.shape)==2:
                input = input.unsqueeze(0)
            output = layer(input, attention_mask=attention_mask,position_ids=position_id, position_embeddings=position_embedding)
            if isinstance(next_layer, nn.Sequential):
                tmp_outputs['layer'] = next_layer(output).detach()
            else:
                tmp_outputs['layer'] = next_layer(output, attention_mask=attention_mask,position_ids=position_id, position_embeddings=position_embedding).detach()
            
            # Clean up intermediate variables
            del input, output

    return tmp_outputs

def make_input_data(data, dev):
    return data.to(dev)

def log_memory_usage(logger, checkpoint_name, dev, block_index=None, epoch=None, step=None, input_shape=None):
    """Log GPU and CPU memory usage at checkpoints"""
    gpu_memory_allocated = torch.cuda.memory_allocated(dev) / 1024**2
    gpu_memory_reserved = torch.cuda.memory_reserved(dev) / 1024**2
    cpu_memory_mb = psutil.virtual_memory().used / 1024**2
    cpu_memory_percent = psutil.virtual_memory().percent
    
    log_msg = f"MEMORY_CHECKPOINT[{checkpoint_name}]"
    if block_index is not None:
        log_msg += f" block:{block_index}"
    if epoch is not None:
        log_msg += f" epoch:{epoch}"
    if step is not None:
        log_msg += f" step:{step}"
    if input_shape is not None:
        log_msg += f" input_shape:{input_shape}"
    log_msg += f" GPU_alloc:{gpu_memory_allocated:.1f}MB GPU_reserved:{gpu_memory_reserved:.1f}MB CPU:{cpu_memory_mb:.1f}MB({cpu_memory_percent:.1f}%)"
    
    logger.info(log_msg)

def train_module(args, modules, module_name, layers, quant_train_inps, fp_train_inps, dtype, dev, position_embeddings, attention_mask_batch, position_ids, logger, block_indices):
    # loss_func = torch.nn.MSELoss(reduction='none')  # Use 'none' to handle padding mask
    loss_func = torch.nn.MSELoss(reduction='mean')  # Use 'none' to handle padding mask
    

    if args.epochs > 0:
        # Convert all modules to float32 for AMP training
        with torch.no_grad():
            for module in modules:
                module.float()
        
        # step 6.3: create optimizer and learning rate schedule for all modules
        param = []
        assert args.quant_lr > 0 or args.weight_lr > 0
        param_group_index = 0
        # Calculate effective training iterations considering gradient accumulation
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        effective_batch_size = args.batch_size * gradient_accumulation_steps
        total_training_iteration = args.epochs * args.train_size / effective_batch_size 
        
        if args.quant_lr > 0:
            # Collect quantization parameters from all modules
            all_quant_params = []
            for module in modules:
                set_quant_parameters(module, True)
                all_quant_params.extend(quant_parameters(module))
            param.append({"params": all_quant_params, "lr": args.quant_lr})
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
            quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
            quant_index = param_group_index
            param_group_index += 1
        else:
            for module in modules:
                set_quant_parameters(module, False)
            
        if args.weight_lr > 0:
            # Collect weight parameters from all modules
            all_weight_params = []
            for module in modules:
                set_weight_parameters(module, True)
                all_weight_params.extend(weight_parameters(module))
            param.append({"params": all_weight_params, "lr": args.weight_lr})
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
            weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
            weight_index = param_group_index
            param_group_index += 1
        else:
            for module in modules:
                set_weight_parameters(module, False)
        optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
        loss_scaler = torch.cuda.amp.GradScaler()
        trainable_parameters = 0
        for module in modules:
            trainable_parameters += trainable_parameters_num(module)
        print(f"trainable parameter number: {trainable_parameters/1e6}M")

        best_val_loss = 1e6
        early_stop_flag = 0
        scale_factor = None
        
        # fp_train_inps already contains ground truth labels after update_dataset calls
        logger.info(f"Using fp_train_inps as ground truth labels (already passed through {len(block_indices)} layers: {block_indices})")
        
        forward_modules = [make_forward(module) for module in modules]
        for epoch in range(args.epochs):

            
            # step: 6.4 training
            loss_list = []
            norm_list = []
            mask_list = []
            start_time = time.time()
            accumulated_loss = 0.0
            
            for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                assert len(quant_inps.shape) == 3, "quant inps should be 3D tensor"
                assert len(fp_inps.shape) == 3, "fp inps should be 3D tensor"
                

                
                # obtain output of quantization model
                with torch.cuda.amp.autocast(dtype=dtype):
                    quant_inps = quant_inps.to(dev)
                    input = make_input_data(quant_inps, dev)
                    fp_inps = fp_inps.to(dev)

                    
                    # Transfer position_embeddings tuple to GPU (once)
                    pos_emb_gpu = tuple(emb.to(dev) for emb in position_embeddings[index])
                    position_ids_gpu = position_ids[index].to(dev)
                    
                    # Use fp_inps as ground truth label (already updated through layers)
                    label_layer = fp_inps

                    # position_idsのバッチサイズをinputに合わせて修正
                    if position_ids_gpu.shape[0] != input.shape[0]:
                        position_ids_gpu = position_ids_gpu.expand(input.shape[0], -1)

                    # Forward through all quantized modules in sequence
                    current_output = input
                    for module in forward_modules:
                        current_output = module(current_output, attention_mask=attention_mask_batch, position_ids=position_ids_gpu, position_embeddings=pos_emb_gpu)
                    
                    out_layer = current_output
                


                layer_reconstruction_loss = loss_func(out_layer.float(), label_layer.float())
                reconstruction_loss = layer_reconstruction_loss
                loss = reconstruction_loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

                if not math.isfinite(loss.item()):
                    raise FloatingPointError("Nonfinite Block-AP loss")
                
                # Accumulate loss for logging (unscaled)
                accumulated_loss += layer_reconstruction_loss.detach().cpu()
                
                # Backward pass (gradients are accumulated)
                loss.backward()

                # Update parameters only when we've accumulated enough gradients
                if (index + 1) % gradient_accumulation_steps == 0 or (index + 1) == len(quant_train_inps):

                    
                    # Gradient clipping (optional) - apply to all modules
                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm > 0:
                        all_trainable_params = []
                        for module in modules:
                            all_trainable_params.extend(trainable_parameters(module))
                        torch.nn.utils.clip_grad_norm_(all_trainable_params, args.max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                    # adjust lr (only when we actually update)
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr > 0:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                
                # Log loss every step (for compatibility with existing logging)
                if (index + 1) % gradient_accumulation_steps == 0 or (index + 1) == len(quant_train_inps):
                    loss_list.append(accumulated_loss / gradient_accumulation_steps)
                    accumulated_loss = 0.0
                

                    
            train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
            loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
            val_loss_mean = 0.0
            # Get CPU memory usage
            cpu_memory_mb = psutil.virtual_memory().used / 1024**2
            cpu_memory_percent = psutil.virtual_memory().percent
            
            logger.info(f"blocks {block_indices} epoch {epoch} recon_loss:{loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} cpu_memory:{cpu_memory_mb:.1f}MB({cpu_memory_percent:.1f}%) time {time.time()-start_time}")
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
            else:
                early_stop_flag += 1
                if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                    break
        optimizer.zero_grad()
        del optimizer

        ## remove for production

    # Preserve training tensors before the export-only FP16 cast and rounding.
    for block_index, module in zip(block_indices, modules):
        args._latent_checkpoint.save_block(block_index, module)

    # step 6.6: directly replace the weight with fake quantization
    for module in modules:
        module.half()
        # quant_inplace(module)
        # set_quant_state(module,weight_quant=False)  # weight has been quantized inplace
    return modules

def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    if args.epochs > 0:
        from OnQ.quantization.runtime_compat import require_compatible_quantization_gradients
        require_compatible_quantization_gradients()
    # Add default nblock parameter if not exists
    if not hasattr(args, 'nblock'):
        args.nblock = 1
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    latent_dir = getattr(args, 'save_latent_dir', None)
    if not latent_dir:
        latent_dir = (str(args.save_quant_dir) + '_latent' if args.save_quant_dir
                      else os.path.join(args.output_dir, 'latent_checkpoint'))
    args._latent_checkpoint = BlockwiseLatentWriter(
        latent_dir, num_blocks=len(layers), bits=args.wbits, group_size=args.group_size)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.bfloat16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    # print cpu memory usage
    print(f"after creating fp_train_inps: {psutil.Process(os.getpid()).memory_info().rss/1024**3:.1f} GB") 
    # step 3: catch the input of thefirst layer 
    return_dict = dict()
    return_dict["position_embeddings"] = []
    return_dict["position_ids"] = []
    index = 0
    def _hook(module, input, kwargs, output):

        nonlocal return_dict
        nonlocal fp_train_inps
        nonlocal index
        fp_train_inps.update_data(index, input[0].to('cpu'))
        index += 1
        return_dict["attention_mask"] = kwargs["attention_mask"]
        # return_dict["position_embeddings"] = kwargs["position_embeddings"]
        # Convert position_embeddings tuple to CPU to save GPU memory
        pos_emb_cpu = tuple(emb.to("cpu") for emb in kwargs["position_embeddings"])
        return_dict["position_embeddings"].append(pos_emb_cpu)
        return_dict["position_ids"].append(kwargs["position_ids"].to("cpu"))
        raise ValueError

    hook = layers[0].register_forward_hook(_hook, with_kwargs=True)

    # step 3.1: catch the input of training set
    # layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            # cpu memory usage
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)

            # # delete trainloader from cpu
            # cpu_memory_mb = psutil.virtual_memory().used / 1024**2
            # cpu_memory_percent = psutil.virtual_memory().percent
            # gpu_memory_mb = torch.cuda.max_memory_allocated(dev) / 1024**2
            # logger.info(f"update_dataset iter {i}: cpu_memory:{cpu_memory_mb:.1f}MB({cpu_memory_percent:.1f}%) gpu_memory:{gpu_memory_mb:.1f}MB")
            try:
                model(data.to(dev))
            except ValueError:
                del data
                pass
    del trainloader
    hook.remove()
    attention_mask = return_dict["attention_mask"]
    position_ids = return_dict["position_ids"]
    position_embeddings = return_dict["position_embeddings"]
    # print(position_embeddings[0][0].shape, len(position_embeddings))
    # exit(0)

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    # step 4: move embedding layer and first layer to cpu
    # layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    # model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        if os.path.isdir(fp_val_cache_path):
            shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)

    # step 6: start training    
    rotary_emb = None
    if hasattr(model.model, 'rotary_emb'):
        rotary_emb = model.model.rotary_emb
        assert rotary_emb is not None, "rotary_emb is not found" 

    # Process blocks in groups of nblock
    for block_start in range(0, len(layers), args.nblock):
        block_end = min(block_start + args.nblock, len(layers))
        block_indices = list(range(block_start, block_end))
        
        logger.info(f"=== Start quantize blocks {block_indices}==={psutil.Process(os.getpid()).memory_info().rss/1024**3:.1f} GB")
        
        # step 6.1: prepare original layers and quantized layers for this group
        original_layers = []
        qlayers = []
        
        for block_index in block_indices:
            layer = layers[block_index].to(dev)
            original_layers.append(layer)
            
            qlayer = copy.deepcopy(layer)
            # Replace Linear with QuantLinear for QAT
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear) or isinstance(module, int_linear_fake.QuantLinear):
                    print(f"replace with QuantLinear: {name}-{module.weight.reshape(-1, args.group_size)[0].unique().numel()}")
                    quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, getattr(args, 'quantizer_class', 'UniformAffineQuantizer'), scale=args.scale)
                    set_op_by_name(qlayer, name, quantlinear)
                    del module
            qlayer.to(dev)
            qlayers.append(qlayer)
            print(qlayer)

        # step 6.2: set quantization states
        for qlayer in qlayers:
            set_quant_state(qlayer, weight_quant=False)  # deactivate quantization for obtaining ground truth
        for qlayer in qlayers:
            set_quant_state(qlayer, weight_quant=True)   # activate quantization

        assert rotary_emb is not None, "rotary_emb is not found"

        # Update dataset sequentially through all layers to get ground truth
        logger.info(f"begin dataset update for blocks {block_indices}")
        for block_index, layer in zip(block_indices, original_layers):
            update_dataset(layer, fp_train_inps, dev, attention_mask, position_ids, position_embeddings=position_embeddings, logger=None)
            logger.info(f"updated dataset through layer {block_index}")
        logger.info(f"end dataset update for blocks {block_indices} - fp_train_inps now contains ground truth")

        # Select input dataset based on args.use_fp_inputs
        train_inps = fp_train_inps if args.use_fp_inputs else quant_train_inps
        logger.info(f"Using {'fp_train_inps' if args.use_fp_inputs else 'quant_train_inps'} for training modules")

        # train all layers in this block group simultaneously
        # Note: fp_train_inps now contains ground truth labels after update_dataset calls
        trained_qlayers = train_module(
            args=args, modules=qlayers, module_name='block_group', layers=original_layers,
            quant_train_inps=quant_train_inps, fp_train_inps=fp_train_inps,
            dtype=dtype, dev=dev, position_embeddings=position_embeddings, attention_mask_batch=attention_mask_batch,
            position_ids=position_ids, logger=logger, block_indices=block_indices
        )

        # Apply quantization inplace and update layers
        for i, (block_index, qlayer) in enumerate(zip(block_indices, trained_qlayers)):
            quant_inplace(qlayer)
            set_quant_state(qlayer, weight_quant=False)
            layers[block_index] = qlayer

        # # Calculate loss for evaluation
        # loss_func = torch.nn.MSELoss(reduction='none')
        # if args.epochs > 0:
        #     for block_index, qlayer in zip(block_indices, trained_qlayers):
        #         losses = []
        #         for index, (label, quant_inp) in enumerate(zip(fp_train_inps, quant_train_inps)):
        #             if index == 32:
        #                 break
        #             with torch.no_grad():
        #                 quant_inp = quant_inp.to(dev)
        #                 label = label.to(dev)
        #                 with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        #                     pos_emb_gpu = tuple(emb.to(dev) for emb in position_embeddings[index])
        #                     output = qlayer(quant_inp, attention_mask=attention_mask, position_ids=position_ids[index].to(dev), position_embeddings=pos_emb_gpu)
        #                 loss = loss_func(output.float(), label.float())
        #                 losses.append(loss.mean().item())
        #         print(f"average loss for layer {block_index}: {sum(losses)/len(losses)}")

        # Update quantized training dataset
        for block_index in block_indices:
            module = layers[block_index]
            update_dataset(module, quant_train_inps, dev, attention_mask, position_ids, position_embeddings=position_embeddings)

        # Move layers to CPU to save GPU memory
        for block_index in block_indices:
            layers[block_index] = layers[block_index].to('cpu')

        # step 7: pack quantized weights into low-bits format
        if args.real_quant:
            for block_index, qlayer in zip(block_indices, trained_qlayers):
                named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                for name, module in named_linears.items():
                    scales = module.weight_quantizer.scale.clamp(1e-4, 1e4).detach()
                    zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                    group_size = module.weight_quantizer.group_size
                    dim0 = module.weight.shape[0]
                    scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                    zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                    q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features, module.out_features, not module.bias is None)
                    q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                    set_op_by_name(qlayer, name, q_linear)
                    logger.info(f"pack quantized {name} finished")
                    del module

        # Clean up
        for layer in original_layers:
            del layer
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model
