from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
import paddle
from sspcab.dataset import MVTecAT
from sspcab.model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle
from sklearn.utils import shuffle
from collections import defaultdict
from sspcab.density import GaussianDensitySklearn, GaussianDensityPaddle
import pandas as pd
from sspcab.utils import str2bool
import os
import warnings
warnings.filterwarnings("ignore")

test_data_eval = None
test_transform = None
cached_type = None


def get_train_embeds(data_dir, model, size, defect_type, transform):
    # train data / train kde
    test_data = MVTecAT(data_dir, defect_type, size, transform=transform, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=64,
                                  shuffle=False, num_workers=0)
    train_embed = []
    with paddle.no_grad():
        for x in dataloader_train:
            embed, logit, _ = model(x)

            train_embed.append(embed.cpu())
    train_embed = paddle.concat(train_embed)
    return train_embed


def eval_model(data_dir, model_dir, defect_type, device="cpu", save_plots=False, size=256, show_training_data=True, model=None, train_embed=None, head_layer=8,
               density=GaussianDensityPaddle()):
    # create test dataset
    global test_data_eval, test_transform, cached_type

    # TODO: cache is only nice during training. do we need it?
    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size, size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT(data_dir, defect_type, size, transform=test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=64,
                                 shuffle=False, num_workers=0)

    model_name = f"model-{defect_type}"
    model_name_dir = Path(model_dir) / f"{model_name}.pdparams"
    # create model
    if model is None:
        print(f"loading model {model_name}")
        head_layers = [512] * head_layer + [128]
        print(head_layers)
        weights = paddle.load(str(model_name_dir))
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_dict(weights)
        model.to(device)
        model.eval()

    # get embeddings for test data
    labels = []
    embeds = []
    with paddle.no_grad():
        for x, label in dataloader_test:
            embed, logit, _ = model(x)

            # save
            embeds.append(embed)
            labels.append(label)
    labels = paddle.concat(labels)
    embeds = paddle.concat(embeds)

    if train_embed is None:
        train_embed = get_train_embeds(data_dir, model, size, defect_type, test_transform)

    # norm embeds
    embeds = paddle.nn.functional.normalize(embeds, p=2, axis=1)
    train_embed = paddle.nn.functional.normalize(train_embed, p=2, axis=1)

    # create eval plot dir
    if save_plots:
        eval_dir = Path("eval") / model_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        # plot tsne
        tsne_labels = labels
        tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")

    print(f"using density estimation {density.__class__.__name__}")
    density.fit(train_embed)
    distances = density.predict(embeds)

    fpr, tpr, threshold = roc_curve(labels, distances)
    right_index = (tpr + (1 - fpr) - 1)
    import numpy as np
    index = np.argmax(right_index)
    threshold = threshold[index]

    if show_training_data:
        dict_train_embed = {'train_embed': train_embed.numpy(),
                            'threshold': threshold}
        train_embed_dir = Path("eval") / f"{model_name}"
        train_embed_dir.mkdir(parents=True, exist_ok=True)
        fpkl = open(os.path.join(train_embed_dir, "dict_train_embed.pkl"), 'wb')
        pickle.dump(dict_train_embed, fpkl)
        fpkl.close()

    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=model_name_dir, save_plots=save_plots)

    return roc_auc


def plot_roc(labels, scores, filename, modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # plot roc
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds, labels.cast("int64"))
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--data_dir', default="data",
                        help='input folder of the models ')

    parser.add_argument('--model_dir', default="models",
                        help=' directory contating models to evaluate (default: models)')

    parser.add_argument('--cuda', default=False, type=str2bool,
                        help='use cuda for model predictions (default: False)')

    parser.add_argument('--head_layer', default=8, type=int,
                        help='number of layers in the projection head (default: 8)')

    parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: paddle)')

    parser.add_argument('--save_plots', default=True, type=str2bool,
                        help='save TSNE and roc plots')

    args = parser.parse_args()

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

    device = "cuda" if args.cuda else "cpu"

    density_mapping = {
        "paddle": GaussianDensityPaddle,
        "sklearn": GaussianDensitySklearn
    }
    density = density_mapping[args.density]

    obj = defaultdict(list)
    for data_type in types:
        print(f"evaluating {data_type}")

        roc_auc = eval_model(args.data_dir, args.model_dir, data_type, save_plots=args.save_plots, device=device, head_layer=args.head_layer, density=density())
        print(f"{data_type} AUC: {roc_auc}")
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)

    # save pandas dataframe
    eval_dir = Path("eval") / args.model_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "_perf.csv")
