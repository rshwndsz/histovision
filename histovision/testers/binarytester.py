import torch
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
import cv2

# Optimizer chosen from cfg.optimizer
from torch import optim
# Dataset chosen from cfg.dataset
import histovision.datasets
# Model chosen from cfg.model
import histovision.models

from .basetester import BaseTester


class BinaryTester(BaseTester):
    def __init__(self, cfg):
        super(BinaryTester, self).__init__()
        self.cfg = cfg
        self.net = hydra.utils.instantiate(self.cfg.model).to(self.cfg.device)
        self.net.eval()
        self.dataloader = eval(self.cfg.provider)('test', self.cfg)

    def forward(self, images):
        images = images.to(self.cfg.device)
        probs = torch.sigmoid(self.net(images))
        preds = (probs > 0.5).float()

        return preds

    def start(self):
        # Load the net
        checkpoint = torch.load(self.cfg.testing.checkpoint_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        # Test the net
        with torch.no_grad():
            for i, images in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                preds = self.forward(images)
                display(images, preds)
                if self.cfg.testing.save_predictions:
                    cv2.imwrite(self.cfg.testing.save_dir + 'pred_' + str(i) + '.png',
                                preds.cpu().detach().numpy().squeeze())


def display(images, preds):
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(preds.cpu().numpy().squeeze(), 'gray')
    ax[1].imshow(images.cpu().numpy().squeeze().transpose(1, 2, 0))
    ax[0].set_title("Predictions")
    ax[1].set_title("Images")
    plt.show()
