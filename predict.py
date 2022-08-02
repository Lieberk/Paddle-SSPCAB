from paddle.vision import transforms
import paddle
from sspcab.model import ProjectionNet
from PIL import Image
from sspcab.density import GaussianDensityPaddle
import argparse
import os
from pathlib import Path
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='use this model.')
parser.add_argument('--cuda', default=True, help='use gpu')
parser.add_argument('--data_type', default='bottle', type=str, help='which data type need to be detected')
parser.add_argument('--img_file', default='images/bad.png', type=str, help='the path of image that will be detected')
parser.add_argument('--model', default='models/', type=str,
                    help='where the params of model saved')
parser.add_argument('--img_size', default=256,
                    help='image size for model infer (default: 256)')
parser.add_argument('--seed', default=102, type=int, help="number of random seed")
args = parser.parse_args()


def load_all_model(model_name, size, head_layer=1, num_classes=3):
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    model_name_real = str(model_name) + '.pdparams'
    weights = paddle.load(model_name_real)
    head_layers = [512] * head_layer + [128]
    model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=num_classes)
    model.set_state_dict(weights)
    model.eval()
    return model, test_transform


def infer(model_name, data_type, size):
    model, transform = load_all_model(model_name, size)
    density = GaussianDensityPaddle()
    img = Image.open(args.img_file).resize((size, size)).convert("RGB")
    img = transform(img)
    img = paddle.unsqueeze(img, axis=0)
    with paddle.no_grad():
        embed, logit, _ = model(img)
    embed = paddle.nn.functional.normalize(embed, p=2, axis=1)
    with open(os.path.join(Path("eval") / f"model-{data_type}", "dict_train_embed.pkl"), "rb") as f:
        info = pickle.load(f)
        train_embed = paddle.to_tensor(info['train_embed'])
        best_threshold = paddle.to_tensor(info['threshold'])
    density.fit(train_embed)
    distances = density.predict(embed)
    if distances[0] >= best_threshold:
        print("预测结果为：异常 预测分数为：%.4f"%distances[0])
    else:
        print("预测结果为：正常 预测分数为：%.4f"%distances[0])


if __name__ == '__main__':
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    model_name = args.model + "model-" + args.data_type
    infer(model_name, args.data_type, args.img_size)
