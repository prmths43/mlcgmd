import os
from pathlib import Path
from typing import Optional, Union, List  # Added Union and List

import dotenv
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict  # Added open_dict


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
           it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


STATS_KEY: str = "stats"

def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - sizes of train, val, test dataset
        - number of trainable model parameters
    Args:
        cfg (DictConfig): [description]
        model (pl.LightningModule): [description]
        trainer (pl.Trainer): [description]
    """
    # Create a mutable copy safely
    with open_dict(cfg):  # Added context manager
        hparams = OmegaConf.to_container(cfg, resolve=True)
    
    # Save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Handle both single logger and multiple loggers
    loggers = trainer.loggers
    if not isinstance(loggers, list):  # Handle single logger case
        loggers = [loggers]
    
    for logger in loggers:  # Loop through all loggers
        logger.log_hyperparams(hparams)
        # Disable future logging for this logger
        logger.log_hyperparams = lambda params: None


# Load environment variables
load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
if PROJECT_ROOT.exists():  # Removed extra parentheses
    os.chdir(PROJECT_ROOT)
