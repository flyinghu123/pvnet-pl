import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger

from src.lightning.lightning_pvnet import LitPvnet
from src.datasets.linemod import LineModDataModule

warnings.filterwarnings("ignore")


def run(cfg: DictConfig) -> None:
    pl.seed_everything(1234, workers=True)  # 设置随机种子
    # torch.multiprocessing.set_sharing_strategy('file_system')
    model = LitPvnet(cfg)
    data_model = LineModDataModule(cfg, os.path.join(hydra.utils.get_original_cwd(), 'data', 'linemod'))
    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)
    trainer = pl.Trainer(
        logger=tb_logger,
        **cfg.trainer,
    )
    trainer.fit(model, data_model)

    # save as a simple torch model
    # model_name = os.getcwd().split('\\')[-1] + '.pth'
    # print(model_name)
    # torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path='conf', config_name='config')
def run_model(cfg: DictConfig) -> None:
    print(cfg)
    # save_useful_info()  # 保存对应代码和配置  shutil.copytree(os.path.join(hydra.utils.get_original_cwd(), 'src'), os.path.join(os.getcwd(), 'code/src'))
    run(cfg)


if __name__ == '__main__':
    run_model()
