import os
import torch
from method_src.data import get_stream_with_prev_dataset
from method_src.dataset import With_GA_StreamDataLoader
from src.models.aion import AION
from method_src.stream_trainer_aion import StreamTrainerAion
from logging import getLogger

logger = getLogger()


def run_stream(args, device):
    # Data
    logger.info("Using AION stream")
    train_stream, test_stream = get_stream_with_prev_dataset(args)
    train_stream_dataloader = With_GA_StreamDataLoader(train_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_stream_dataloader = With_GA_StreamDataLoader(test_stream, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    groups = 3
    pretrained_model = AION(args, groups=groups).to(device)
    pretrained_model.load_state_dict(torch.load(os.path.join(args.model_save_pth, f"AION_pretrain_{args.seed}_{args.pretrain_remarks}.pth"), map_location=device))
    logger.info(f"pretrained_model: AION_pretrain_{args.seed}_{args.pretrain_remarks}.pth")
    logger.info(f"Pretrained Model:\n {pretrained_model}")

    # Trainer
    stream_trainer = StreamTrainerAion(args, pretrained_model, device, train_stream_dataloader, test_stream_dataloader)
    res = stream_trainer.train()
    return res
