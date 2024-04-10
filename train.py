import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
import wandb
from model import GPTConfig, GPT, MoEGPT
import hydra
from utils import get_dataloader, get_dataloader_lol, dotdict, get_cosine_warmp_lr, check_generated_path_accuracy
import pdb
from init import set_seed, open_log, init_wandb, cleanup

if torch.cuda.is_available():
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
else:
    dtype = 'None'

# HYDRA_FULL_ERROR=1

@hydra.main(config_path="./configs", config_name="test_config.yaml")
def main(cfg):

    # Initialize wandb project
    init_wandb(cfg, cfg.wandb_project)

    set_seed(cfg.seed)
    # create log file
    fp = open_log(cfg)
    # decide whether cpu or gpu
    device = cfg.device

    tokens_per_iter = cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size

        # Convert the string to an integer or float
    tokens_per_iter = int(tokens_per_iter)  # or float(tokens_per_iter)

        # Use the ',' format specifier to include commas as thousands separators
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
 

    os.makedirs(cfg.out_dir, exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'None': torch.float32}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    
    ### DATA STUFF HERE
    data_dir_train = cfg.dataset_path + 'tokens_path_train.npy'
    data_dir_eval = cfg.dataset_path + 'tokens_path_eval.npy'
    # Get the number of unique tokens in the dataset (meta_vocab_size) from the training data file to initialize model
    data = np.load(data_dir_train, allow_pickle=True)
    flattened_data = [element for sublist in data for element in sublist]
    meta_vocab_size = len(list(set(flattened_data)))

    #Create dataloaders
    train_dataloader, val_dataloader = get_dataloader_lol(train_data_path=data_dir_train, val_data_path=data_dir_eval, batch_size = cfg.batch_size, num_workers=1)

    def pick_dataloader(split):
        dataloader = train_dataloader if split == 'train' else val_dataloader
        return dataloader

    # Load DAG and token_map to check paths:
    path = cfg.dataset_path 
    scm_file_path = path + 'random_frozen_scm_used.npz'
    scm_dict = np.load(scm_file_path, allow_pickle=True)
    with open(path + 'dag_path_scm.pkl', "rb") as f:
        dag =  pickle.load(f)
    token_map = scm_dict['token_map'].item()

    # model init
    # add + 1 to meta_vocab_size to account for padding token with TOKENID=0
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size, bias=cfg.bias, vocab_size=meta_vocab_size+1, dropout=cfg.dropout)
                    
    if cfg.init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # gptconf = GPTConfig(**model_args)
        # model = GPT(gptconf)

        cfg.vocab_size = meta_vocab_size + 1
        model = MoEGPT(cfg)

   # Checkpointing
    elif cfg.init_from == 'resume':
        print(f"Resuming training from {cfg.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt9950.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # crop down the model block size if desired, using model surgery
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args['block_size'] = cfg.block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    #Mixed precision training: look at gradients convert 32 bits to 16 bits
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(optimizer=cfg.optimizer, weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), device_type=device_type)
    if cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0


    # helps estimate an arbitrarily accurate loss over either split using many batches only for eval:
    @torch.no_grad()
    def estimate_loss():
        '''
        Return a dictionary containing train loss and val loss
        '''

        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
    
            dataloader = iter(pick_dataloader(split))
            for k in range(cfg.eval_iters):
                try:
                    X,Y = next(dataloader)
                except StopIteration:
                    dataloader = iter(pick_dataloader('train'))
                    X,Y = next(dataloader)

                if device_type == 'cuda':
                    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                    X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
                else:
                    X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)

                losses[k] = loss.item()
                
            out[split] = losses.mean()
        model.train()
        return out

    # training loop
    dataloader = iter(pick_dataloader('train')) # fetch the very first batch of X,Y
    X,Y = next(dataloader)
    X, Y = X.to(device), Y.to(device)

    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model =  model 
    running_mfu = -1.0

    print('Train loop started')

    while True:

        # determine and set the learning rate for this iteration
        lr = get_cosine_warmp_lr(iter_num, cfg.learning_rate, cfg.warmup_iters, cfg.lr_decay_iters, cfg.min_lr) if cfg.decay_lr else cfg.learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0:# and master_process:
    
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to wandb
            if cfg.deploy:

                ## Evaluate accuracy of model
                top_k = cfg.top_k
                temperature = cfg.temperature
                max_new_tokens = cfg.block_size
                num_samples = cfg.num_samples_generated_for_accuracy
                dataloader = iter(val_dataloader)
                n = 3
                generated_paths = []
                model.eval()
                with torch.no_grad():
                    with ctx:
                        for k in range(num_samples):
                            x,_ = next(dataloader)
                            # Generate a list of lists of sequences
                            # Each sublist of size batch_size x max_new_tokens + n
                            generated_paths.append(model.generate(x[0:,0:n].to(device), max_new_tokens, temperature=temperature, top_k=top_k))
                model.train()
                edge_accuracies, does_end_at_targets, path_lengths = check_generated_path_accuracy(dag, generated_paths, token_map)
                edge_accuracies[np.isnan(edge_accuracies)] = 0
                

                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "edge_accuracies": np.mean(edge_accuracies),
                    "does_end_at_target": np.mean(does_end_at_targets),
                    "path_lengths": np.mean(path_lengths)
                })

            # evaluate and checkpoint model
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                
                if iter_num > 0 and iter_num%(cfg.save_ckpt_interval)==0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }

                    #Save checkpoint with iter num:
                    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt' + str(iter_num) + '.pt'))
                    print(f"saving checkpoint to {cfg.out_dir}")

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(cfg.gradient_accumulation_steps):

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            try:
                X,Y = next(dataloader)
            except StopIteration:
                dataloader = iter(pick_dataloader('train'))
                X, Y = next(dataloader)

            if device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)

            else:
                X, Y = X.to(device), Y.to(device)
                
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * cfg.gradient_accumulation_steps

        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > cfg.max_iters:
            break



if __name__ == "__main__":
    main()




