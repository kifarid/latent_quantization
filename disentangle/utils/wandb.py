import omegaconf
import wandb
import os
import hydra


def initialize_wandb(config, name):
    #if debug is true then offline mode
    run = wandb.init(
        project=config.wandb.project,
        config=omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        save_code=True,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        name=config.wandb.name if config.wandb.name is not None else name,
        mode="offline" if config.debug or config.wandb.offline else "online",


    )
    wandb.config.update({'wandb_run_dir': wandb.run.dir})
    wandb.config.update({'hydra_run_dir': os.getcwd()}, allow_val_change=True)
    return run