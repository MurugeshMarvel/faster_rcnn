import argparse
import os
import time
from collections import deque
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.base_model import BaseModel
from utils.dataset import DataSet
from model import Model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _train(backbone_name, path_to_data_dir, path_to_checkpoints_dir):
    dataset = DataSet(path_to_data_dir, mode=DataSet.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    backbone = BaseModel.select_model(backbone_name)(pretrained=True)
    model = Model(backbone).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=50000, gamma=0.1)

    step = 0
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    num_steps_to_display = 20
    num_steps_to_snapshot = 10000
    num_steps_to_stop_training = 70000

    print('Start training')

    while not should_stop:
        for batch_index, (_, image_batch, _, bboxes_batch, labels_batch) in enumerate(dataloader):
            assert image_batch.shape[0] == 1, 'only batch size of 1 is supported'

            image = image_batch[0].to(DEVICE)
            bboxes = bboxes_batch[0].to(DEVICE)
            labels = labels_batch[0].to(DEVICE)

            forward_input = Model.ForwardInput.Train(image,gt_classes =labels, gt_bboxes=bboxes)
            forward_output = model.train().forward(forward_input)

            loss = forward_output.anchor_objectness_loss + forward_output.anchor_transformer_loss + \
                forward_output.proposal_class_loss + forward_output.proposal_transformer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            step += 1

            if step % num_steps_to_display == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = num_steps_to_display / elapsed_time
                avg_loss = sum(losses) / len(losses)
                lr = scheduler.get_lr()[0]
                print('[Step {}] Avg. Loss = {}, Learning Rate = {} ({} steps/sec)'.format(step,avg_loss,lr,steps_per_sec))

            if step % num_steps_to_snapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print('Model saved to {}'.format(path_to_checkpoint))

            if step == num_steps_to_stop_training:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--backbone', choices=['vgg16', 'resnet101'], required=True, help='name of backbone model')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        backbone_name = args.backbone
        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        os.makedirs(path_to_checkpoints_dir, exist_ok=True)

        _train(backbone_name, path_to_data_dir, path_to_checkpoints_dir)

    main()
