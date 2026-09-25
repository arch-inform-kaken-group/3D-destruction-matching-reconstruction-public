"""
Inference on the held-out Jomon objects.
Loads base GARF.ckpt, applies the fine-tuned LoRA checkpoint, then runs
two-session flow matching via trainer.test (writes json_results/).

Run from GARF/ root:
  python infer_jomon.py experiment=jomon_finetune \
      data.data_root=/abs/path/jomon_data \
      +base_ckpt_path=output/GARF.ckpt \
      ckpt_path=logs/GARF-Jomon/<run>/checkpoints/last.ckpt \
      ++model.inference_config.one_step_init=true \
      trainer.devices=[0]
"""
from typing import List

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])


@hydra.main(version_base="1.3", config_path="./configs", config_name="eval")
def main(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    loggers: List[Logger] = [
        hydra.utils.instantiate(logger) for logger in cfg.get("loggers", dict()).values()
    ]
    for logger in loggers:
        logger.log_hyperparams(OmegaConf.to_object(cfg))

    model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.get("data"))
    callbacks: List[L.Callback] = [
        hydra.utils.instantiate(callback) for callback in cfg.get("callbacks").values()
    ]

    # 1) Load base GARF weights (PTv3 + FM bundled in GARF.ckpt)
    base_sd = torch.load(cfg.get("base_ckpt_path"), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(base_sd, strict=False)

    # 2) Apply fine-tuned LoRA weights (cfg.ckpt_path = the LoRA checkpoint)
    model.enable_lora(ckpt_path=cfg.get("ckpt_path"))

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.get("trainer"), callbacks=callbacks, logger=loggers
    )
    # ckpt_path=None -> use the model we just loaded (base + LoRA)
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()