import logging
import torch
import argparse

from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
from core.adapter.dynamic_rotta import DynamicMemorySizeRoTTA


def testTimeAdaptation(cfg):
    logger = logging.getLogger("TTA.test_time")

    # 构建模型和优化器
    model = build_model(cfg)
    optimizer = build_optimizer(cfg)

    # 动态加载适配器
    tta_model = DynamicMemorySizeRoTTA(cfg, model, optimizer)

    tta_model.cuda()

    # 构建数据加载器
    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)

    # 日志记录
    logger.info(f"Using adapter: {cfg.ADAPTER.NAME}")

    # 适配测试循环
    tbar = tqdm(loader)
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']
        if len(label) == 1:
            continue  # 忽略单样本数据
        data, label = data.cuda(), label.cuda()

        # 调用适配器的 forward 方法
        output = tta_model(data)  # 传递数据到适配器
        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        processor.process(accurate, domain)

        if batch_id % 10 == 0:
            if hasattr(tta_model, "mem"):  # 如果适配器有内存占用功能
                tbar.set_postfix(acc=processor.cumulative_acc(), bank=tta_model.mem.get_occupancy())
            else:
                tbar.set_postfix(acc=processor.cumulative_acc())

    # 计算最终结果
    processor.calculate()
    logger.info(f"All Results\n{processor.info()}")


def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(f"Loaded configuration file: \n"
                f"\tadapter: {args.adapter_config_file}\n"
                f"\tdataset: {args.dataset_config_file}\n"
                f"\torder: {args.order_config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    testTimeAdaptation(cfg)


if __name__ == "__main__":
    main()