"""
Main Training Script
"""
from datetime import datetime
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mmppt.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from mmppt.engines.train import TRAINERS
from mmppt.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    if "save_path" in args.options:
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.options["save_path"] = os.path.join(args.options["save_path"], date_time_str)
        if not os.path.exists(args.options["save_path"]):
            os.mkdir(args.options["save_path"])
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
