# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from tqdm import tqdm
import argparse

import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay
import paddle.vision.transforms as transforms
import numpy as np
from sspcab.dataset import MVTecAT, Repeat
from sspcab.cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from sspcab.model import ProjectionNet
from eval import eval_model
from sspcab.utils import str2bool
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")


class Logger(object):
    def __init__(self, filename="Default.txt"):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


def run_training(data_type="screw",
                 data_dir="data",
                 model_dir="models",
                 epochs=256,
                 test_epochs=10,
                 freeze_resnet=20,
                 batch_size=64,
                 cutpate_type=CutPasteNormal,
                 min_scale=1,
                 device="cuda",
                 workers=8,
                 size=256):
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used

    # TODO: use f strings also for the date LOL
    model_name = f"model-{data_type}"

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size, size)))
    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))

    train_data = MVTecAT(data_dir, data_type, transform=train_transform, size=int(size * (1 / min_scale)))
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, use_shared_memory=True)

    if not os.path.exists(f'logdirs/{model_name}'):
        os.makedirs(f'logdirs/{model_name}')
    logger = Logger(f'logdirs/{model_name}/train.log')

    loss_fn = paddle.nn.CrossEntropyLoss()
    batch_cost = 1.0

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out

    max_roc = -np.inf
    dataloader_inf = get_data_inf()
    reader_start = time.time()
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    for step in tqdm(range(epochs)):

        epoch = int(step / 1)
        if epoch == freeze_resnet:
            model.unfreeze()

        batch_embeds = []
        batch_idx, data = next(dataloader_inf)
        train_reader_cost = time.time() - reader_start
        xs = [x for x in data]

        # zero the parameter gradients
        train_start = time.time()
        optimizer.clear_grad()

        xc = paddle.concat(xs, axis=0)
        embeds, logits, cost_sspcab = model(xc)

        # calculate label
        y = paddle.arange(len(xs))
        y = y.repeat_interleave(xs[0].shape[0])
        loss = loss_fn(logits, y) + 0.01 * cost_sspcab

        # regulize weights:
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        train_run_cost = time.time() - train_start

        # save embed for validation:
        if test_epochs > 0 and epoch % test_epochs == 0:
            batch_embeds.append(embeds.cpu())

        log = "Type : {} Train [ Epoch {}/{} ], loss: {:.4f}, avg_reader_cost: {:.4f} " \
              "avg_batch_cost: {:.4f}, " \
              "avg_ips: {:.4f}.".format(
            data_type, epoch + 1, epochs,
            loss.item(), train_reader_cost / batch_cost,
                       (train_reader_cost + train_run_cost) / batch_cost,
                       batch_size / (train_reader_cost + train_run_cost))

        print(log)
        reader_start = time.time()

        # run tests
        if test_epochs > 0 and (epoch + 1) % test_epochs == 0:
            # run auc calculation
            model.eval()
            roc_auc = eval_model(data_dir, model_name, data_type, device=device,
                                 save_plots=False,
                                 size=size,
                                 show_training_data=False,
                                 model=model)
            model.train()
            if max_roc <= roc_auc:
                max_roc = roc_auc
                paddle.save(model.state_dict(), os.path.join(model_dir, f"{model_name}.pdparams"))
            log = "Type : {} Val [ Epoch {}/{} ] max_auc: {:.4f} roc_auc : {:4f}.".format(
                data_type, epoch + 1, epochs, max_roc, roc_auc)
            print(log)
    logger.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--data_dir', default="/home/aistudio/data/",
                        help='input folder of the models ')

    parser.add_argument('--epochs', default=500, type=int,
                        help='number of epochs to train the model , (default: 256)')

    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')

    parser.add_argument('--logs_dir', default="logdirs",
                        help='logs folder of the models ')

    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')

    parser.add_argument('--test_epochs', default=10, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')

    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=96, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize ('
                             'dafault: "64")')

    parser.add_argument('--head_layer', default=1, type=int,
                        help='number of layers in the projection head (default: 1)')

    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'], help='cutpaste variant to use (dafault: "3way")')

    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='use cuda for training (default: False)')

    parser.add_argument("--seed", type=int, default=102)

    parser.add_argument("--min_scale", type=int, default=1)

    parser.add_argument('--weight_decay', type=float, default=0.00003,
                        help='weight_decay')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum used for sgd')

    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    all_types = ['bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper']

    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")

    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion}
    variant = variant_map[args.variant]

    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")

    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for data_type in types:
        print(f"training {data_type}")

        # create Model:
        head_layers = [512] * args.head_layer + [128]
        num_classes = 2 if variant is not CutPaste3Way else 3
        model = ProjectionNet(pretrained=args.pretrained, head_layers=head_layers, num_classes=num_classes)

        if args.freeze_resnet > 0 and args.pretrained:
            model.freeze_resnet()

        if args.optim == "sgd":
            scheduler = CosineAnnealingDecay(learning_rate=args.lr, T_max=10, last_epoch=args.epochs)
            optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=scheduler,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == "adam":
            optimizer = optim.Adam(parameters=model.parameters(),
                                   learning_rate=args.lr, weight_decay=args.weight_decay)
        else:
            print(f"ERROR unkown optimizer: {args.optim_name}")

        run_training(data_type=data_type,
                     data_dir=args.data_dir,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     batch_size=args.batch_size,
                     device=device,
                     cutpate_type=variant,
                     min_scale=args.min_scale,
                     workers=args.workers)
