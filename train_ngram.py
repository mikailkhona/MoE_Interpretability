import os
import pickle
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import wandb
from model import GPTConfig, GPT, MoEGPT
import hydra
from utils import *
from ngram.ngram import *
import pdb

dtype = 'None'
# if torch.cuda.is_available():
#     dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
# else:
#     dtype = 'None'
# HYDRA_FULL_ERROR=1

@hydra.main(version_base=None,config_path="./configs", config_name="ngram_config.yaml")
def main(cfg):
    ### INITIALIZE ###

    set_seed(cfg.seed)  # set random seed
    fp = open_log(cfg) # create log file and initialize wandb
    device = cfg.device # decide whether cpu or gpu
    tokens_per_iter = int(cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, 'checkpoints'), exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'None': torch.float32}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    iter_num = 0
    best_val_loss = 1e9
    
    ### LOAD TRAINING AND EVAL DATA ###

    data_dir_train = cfg.dataset_path + 'train_seqs.npy'
    data_dir_eval = cfg.dataset_path + 'val_seqs.npy'
    # Get the number of unique tokens in the dataset (meta_vocab_size) from the training data file to initialize model
    data = np.load(data_dir_train, allow_pickle=True)
    data_eval = np.load(data_dir_eval, allow_pickle=True)
    print(f'Loaded {len(data)} training seqs and {len(data_eval)} validation seqs')
    flattened_data = [token for path in data for token in path]
    meta_vocab_size = len(list(set(flattened_data)))
    #Create dataloaders
    train_dataloader, val_dataloader = get_dataloader_lol(train_data_path=data_dir_train, val_data_path=data_dir_eval, batch_size=cfg.batch_size, num_workers=1)
    def pick_dataloader(split):
        dataloader = train_dataloader if split == 'train' else val_dataloader
        return dataloader

    # Load multi_ngram model and token_map to check accuracy:
    token_map = np.load(cfg.dataset_path + 'token_map.npz')
    with open(cfg.dataset_path + 'multi_ngram.pkl', "rb") as f:
        multi_ngram = pickle.load(f)

    vocab = multi_ngram.vocab
    context_size = multi_ngram.k
    ngrams = multi_ngram.ngrams

    ### INITIALIZE MODEL ###
    
    # add + 1 to meta_vocab_size to account for padding token with TOKENID=0
    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size, bias=cfg.bias, vocab_size=meta_vocab_size+1, dropout=cfg.dropout)
    
    print("Initializing a new model from scratch")
    cfg.vocab_size = meta_vocab_size + 1
    model = MoEGPT(cfg)
        
    # crop down the model block size if desired, using model surgery
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args['block_size'] = cfg.block_size
    model.to(device)

    ### OPTIMIZATION STUFF###

    # initialize a GradScaler. If enabled=False scaler is a no-op
    # Mixed precision training: look at gradients convert 32 bits to 16 bits
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    optimizer = model.configure_optimizers(optimizer=cfg.optimizer, weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), device_type=device_type)
    checkpoint = None # free up memory

    ### TRAINING ###
    @torch.no_grad()
    def eval_model():        
        '''
        Returns a list containing:
         1. Dict containing train loss and val loss 
         2. List with total expert loads.
         3. Dict with KL divergence of prob. distribution of the model from the correct one for each n-gram, and for the total multi n-gram.
        '''
        mean_losses = {}
        loads = np.zeros(cfg.num_experts)
        model.eval()
        # Losses
        for split in ['train', 'val']:
            kl_divs = []
            losses = torch.zeros(cfg.eval_iters)
            dataloader = iter(pick_dataloader(split))
            for k in range(cfg.eval_iters):
                try:
                    X, Y = next(dataloader) # X and Y are (batch_size x path_tokens)
                except StopIteration:
                    print('Not enough data in validation, using train data instead')
                    dataloader = iter(pick_dataloader('train'))
                    X, Y = next(dataloader)

                if device_type == 'cuda':
                    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                    X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
                else:
                    X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y) # logits is (batch_size x path_tokens x vocab_size)
                    # Expert activations
                    expert_activations = model.transformer.h[0].moe.expert_activations.detach().cpu().numpy() # (batch, num_experts)
                    loads += expert_activations.sum(0)
                    if split == 'val':
                        for batch in range(X.shape[0]):
                            model_log_probs = torch.nn.functional.log_softmax(logits[batch], dim=-1).cpu()
                            seq = X[batch].cpu()
                            for i in range(len(seq)):
                                context = seq[max(0, i-context_size):i].tolist() # context_size size sequence prior to token i
                                context_str = [vocab[token-1] for token in context if token != 0] # convert to string
                                prob_dist = multi_ngram.prob_dist(context_str)
                                # true_probs[0] is zero because the padding token is not in the vocab
                                true_probs = torch.zeros(cfg.vocab_size)
                                true_probs[1:] = torch.tensor([prob_dist[token] for token in vocab])
                                kl_div = torch.nn.functional.kl_div(model_log_probs[i], true_probs, reduction='batchmean').item()
                                kl_divs.append(kl_div)
                losses[k] = loss.item()
            mean_losses[split] = losses.mean()
            if split == 'val': mean_kl_div = sum(kl_divs)/len(kl_divs)
        loads = loads/loads.sum(0) # normalize
        model.train()
        return mean_losses, loads, mean_kl_div

    # Training loop starts
    dataloader = iter(pick_dataloader('train'))
    X, Y = next(dataloader)
    X, Y = X.to(device), Y.to(device)

    while True:
        # determine and set the learning rate for this iteration
        lr = get_cosine_warmp_lr(iter_num, cfg.learning_rate, cfg.warmup_iters, cfg.lr_decay_iters, cfg.min_lr) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.eval_interval == 0:
            # print('evaluating model')
            losses, loads, kl_div = eval_model()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log expert load
            print(f"experts load %: {[round(100*load) for load in loads]}")

            print(f"KL divergence: {kl_div:.4f}")

            if cfg.deploy: # wandb logging
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "KL divergence": kl_div,
                    "loads/exp0": loads[0],
                    "loads/exp1": loads[1],
                    "loads/exp2": loads[2],
                })
            # evaluate and checkpoint model
            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0 and iter_num % cfg.save_ckpt_interval == 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    print(f"saving checkpoint to {cfg.out_dir}")
                    torch.save(checkpoint, os.path.join(cfg.out_dir, 'checkpoints', 'ckpt' + str(iter_num) + '.pt'))

        if iter_num == 0 and cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / cfg.gradient_accumulation_steps

        try:
            X, Y = next(dataloader)
        except StopIteration:
            dataloader = iter(pick_dataloader('train'))
            X, Y = next(dataloader)

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
        else:
            X, Y = X.to(device), Y.to(device)
            
        scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0: # clip gradient
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1

        if iter_num > cfg.max_iters: # termination conditions
            break

    cleanup(cfg, fp)

if __name__ == "__main__":
    main()