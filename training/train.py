# Warsaw University of Technology
# Train MinkLoc model

import argparse
import torch

from training.trainer import do_train
from misc.utils import TrainingParams


if __name__ == '__main__':
    # 直接指定参数值
    config_path = '../config/config_baseline.txt'
    model_config_path = '../models/minkloc3dv2.txt'
    debug_mode = False

    parser = argparse.ArgumentParser(description='Train MinkLoc3Dv2 model')
    parser.add_argument('--config', type=str, default=config_path, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, default=model_config_path,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--debug', dest='debug', action='store_true', default=debug_mode, help='Enable debug mode')

    args = parser.parse_args()
    print('训练参数路径: {}'.format(args.config))
    print('模型参数路径: {}'.format(args.model_config))
    print('Debug模式是否开启: {}'.format(args.debug))

    params = TrainingParams(args.config, args.model_config, debug=args.debug)
    params.print()

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    do_train(params)
