import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from graphwm.data.datamodule import worker_init_fn
from graphwm.data.utils import dict_collate_fn
from graphwm.common import PROJECT_ROOT
from graphwm.model import GNS, PnR

MODELS = {
    'gns': GNS,
    'pnr': PnR
}

def run_eval(cfg):
    seed_everything(cfg.random_seed, workers=True)
    
    model_dir = Path(cfg.model_dir)
    dataclass, modelclass = model_dir.name.split('_')[:2]
    save_dir = Path(cfg.save_dir)
    
    if modelclass == 'pnr':
        folder_name = f'nsteps{cfg.ld_kwargs.step_per_sigma}_stepsize_{cfg.ld_kwargs.step_size}'
    else:
        folder_name = 'rollouts'
        
    rollout_path = save_dir / folder_name / f'seed_{cfg.random_seed}.pt'
    if rollout_path.exists():
        print('Rollout already exists.')
        return

    last_ckpt = model_dir / 'last.ckpt'
    if last_ckpt.exists():
        ckpt_path = str(last_ckpt)
    else:
        ckpts = list(model_dir.glob('*.ckpt'))
        ckpt_epochs = np.array([int(ckpt.name.split('-')[0].split('=')[1]) for ckpt in ckpts])
        ckpt_path = str(ckpts[ckpt_epochs.argmax()])

    print(f'Loading checkpoint: {ckpt_path}')
    model = MODELS[modelclass].load_from_checkpoint(ckpt_path)

    dataset = hydra.utils.instantiate(
        cfg.data,
        seq_len=model.hparams.seq_len,
        dilation=model.hparams.dilation,
        grouping=model.hparams.cg_level,
    )
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        collate_fn=dict_collate_fn,
        persistent_workers=True  # recommended for PyTorch >=1.7 when num_workers>0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    outputs = []

    # Update ld_kwargs in hparams for pnr model
    if modelclass == 'pnr':
        model.hparams.step_per_sigma = cfg.ld_kwargs.step_per_sigma
        model.hparams.step_size = cfg.ld_kwargs.step_size

    start_time = time.time()
    last_component = 0
    last_cluster = 0

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if idx == cfg.num_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            simulate_steps = cfg.rollout_length // model.hparams.dilation
            output = model.simulate(
                batch,
                simulate_steps - model.hparams.seq_len,
                save_positions=cfg.save_pos,
                deter=cfg.deter,
                save_frequency=cfg.save_frequency
            )
            # Detach batch tensors and move to CPU
            output.update({k: v.detach().cpu() for k, v in batch.items()})

            if model.hparams.cg_level > 1:
                output['cluster'] += last_cluster  # fix offsets for cluster
                last_cluster = output['cluster'].max().item() + 1
                if 'component' in output:
                    output['component'] += last_component
                    last_component = output['component'].max().item() + 1
            outputs.append(output)

    elapsed = time.time() - start_time

    # Concatenate tensors from all batches
    outputs = {k: torch.cat([d[k] for d in outputs]) for k in outputs[-1].keys()}
    outputs['time_elapsed'] = elapsed
    outputs['model_params'] = model.hparams
    outputs['eval_cfg'] = cfg

    os.makedirs(save_dir / folder_name, exist_ok=True)
    torch.save(outputs, rollout_path)

    print(f'Finished {cfg.batch_size * cfg.num_batches} rollouts of {cfg.rollout_length} steps.')

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval")
def main(cfg):
    run_eval(cfg)

if __name__ == "__main__":
    main()
