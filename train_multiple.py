import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import wandb
from model import GPTConfig, GPT, MoEGPT
import hydra
from utils import get_dataloader_lol, get_cosine_warmp_lr, check_generated_path_accuracy
from init import set_seed, open_log, init_wandb, cleanup

if torch.cuda.is_available():
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
else:
    dtype = 'None'

@hydra.main(version_base=None,config_path="./configs", config_name="test_config.yaml")
def main(cfg):
    init_wandb(cfg)
    set_seed(cfg.seed)
    fp = open_log(cfg)
    device = cfg.device
    tokens_per_iter = int(cfg.gradient_accumulation_steps * cfg.batch_size * cfg.block_size)
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(cfg.out_dir, exist_ok=True)

    # torch.manual_seed(1337) # didn't we already set the seed?
    if 'cuda' in device:
        device_type = 'cuda'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device_type = 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'None': torch.float32}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    iter_num = 0
    best_val_loss = 1e9
    
    path = cfg.dataset_path
    data_dir_train = path + 'tokens_path_train.npy'
    data_dir_eval = path + 'tokens_path_eval.npy'
    data = np.load(data_dir_train, allow_pickle=True)
    flattened_data = [token for path in data for token in path]
    meta_vocab_size = len(list(set(flattened_data)))

    train_dataloader, val_dataloader = get_dataloader_lol(train_data_path=data_dir_train, val_data_path=data_dir_eval, batch_size=cfg.batch_size, num_workers=1)

    def pick_dataloader(split):
        dataloader = train_dataloader if split == 'train' else val_dataloader
        return dataloader

    scm_file_path = path + 'graph_path.npz'
    scm_dict = np.load(scm_file_path, allow_pickle=True)
    with open(path + 'dag_path.pkl', "rb") as f:
        dag_dict = pickle.load(f)
    token_map = scm_dict['token_map'].item()
    token_idx_map = scm_dict['token_idx_map'].item()
    num_graphs = len(dag_dict)

    model_args = dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd, block_size=cfg.block_size, bias=cfg.bias, vocab_size=meta_vocab_size+1, dropout=cfg.dropout)
                    
    if cfg.init_from == 'scratch':
        print("Initializing a new model from scratch")
        cfg.vocab_size = meta_vocab_size + 1
        model = MoEGPT(cfg)
    elif cfg.init_from == 'resume':
        print(f"Resuming training from {cfg.out_dir}")
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt9950.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args['block_size'] = cfg.block_size
    model.to(device)

    scaler = torch.GradScaler(device_type, enabled=(dtype == 'float16')) if device_type == 'cuda' else None
    optimizer = model.configure_optimizers(optimizer=cfg.optimizer, weight_decay=cfg.weight_decay, learning_rate=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), device_type=device_type)
    if cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None

    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.eval_iters)
            dataloader = iter(pick_dataloader(split))
            for k in range(cfg.eval_iters):
                try:
                    X, Y = next(dataloader)
                except StopIteration:
                    dataloader = iter(pick_dataloader('train'))
                    X, Y = next(dataloader)

                if device_type == 'cuda':
                    X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
                else:
                    X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)

                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    dataloader = iter(pick_dataloader('train'))
    X, Y = next(dataloader)
    X, Y = X.to(device), Y.to(device)

    local_iter_num = 0
    raw_model = model
    running_mfu = -1.0

    print('Train loop started')
    
    expert_activations = {graph_idx: np.zeros(cfg.num_experts) for graph_idx in range(num_graphs)}
    
    while True:
        lr = get_cosine_warmp_lr(iter_num, cfg.learning_rate, cfg.warmup_iters, cfg.lr_decay_iters, cfg.min_lr) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if cfg.deploy:
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
                            x, _ = next(dataloader)
                            generated_paths.append(model.generate(x[0:, 0:n].to(device), max_new_tokens, temperature=temperature, top_k=top_k))
                model.train()
                edge_accuracies, does_end_at_targets, path_lengths = check_generated_path_accuracy(dag_dict, generated_paths, token_map)
                edge_accuracies[np.isnan(edge_accuracies)] = 0

                expert_load = model.transformer.h[0].mlp.load.detach().cpu().numpy()
                print("expert load", expert_load)
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "edge_accuracies": np.mean(edge_accuracies),
                    "does_end_at_target": np.mean(does_end_at_targets),
                    "path_lengths": np.mean(path_lengths),
                    "expert_load": expert_load,
                })

                # Log the expert activations for each graph to wandb
                for graph_idx in range(num_graphs):
                    wandb.log({f"expert_activations/graph_{graph_idx}": expert_activations[graph_idx]}, step=iter_num)

                # Reset the expert activations for the next interval
                expert_activations = {graph_idx: np.zeros(cfg.num_experts) for graph_idx in range(num_graphs)}

            if losses['val'] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                
                if iter_num > 0 and iter_num % cfg.save_ckpt_interval == 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt' + str(iter_num) + '.pt'))
                    print(f"saving checkpoint to {cfg.out_dir}")

        if iter_num == 0 and cfg.eval_only:
            break

        for micro_step in range(cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps

            # Get the graph indices from the current batch
            graph_indices = X[:, 1, 0].cpu().numpy()  # Assuming the second token in each sequence represents the graph index

            # Extract the expert activations from the model
            expert_acts = model.transformer.h[0].mlp.expert_act.detach().cpu().numpy()

            # Update the expert activations for each graph
            for graph_idx, expert_act in zip(graph_indices, expert_acts):
                expert_activations[graph_idx] += expert_act

            try:
                X, Y = next(dataloader)
            except StopIteration:
                dataloader = iter(pick_dataloader('train'))
                X, Y = next(dataloader)

            if device_type == 'cuda':
                X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
            else:
                X, Y = X.to(device), Y.to(device)
                
            scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.log_interval == 0:
            lossf = loss.item() * cfg.gradient_accumulation_steps

        iter_num += 1
        local_iter_num += 1

        if iter_num > cfg.max_iters:
            break

    cleanup(cfg, fp)

if __name__ == "__main__":
    main()