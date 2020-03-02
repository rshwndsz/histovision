import torch
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
from pathlib import Path

# Model chosen from cfg.model
import histovision.models

from .basetester import BaseTester


class BinaryTester(BaseTester):
    def __init__(self, cfg):
        super(BinaryTester, self).__init__()
        self.cfg = cfg
        self.net = hydra.utils.instantiate(self.cfg.model).to(self.cfg.device)
        self.net.eval()
        self.dataloader = hydra.utils.get_method(cfg.provider)('test', self.cfg)

    def forward(self, images):
        images = images.to(self.cfg.device)
        probs = self.net(images).softmax(dim=1)
        preds = probs.argmax(dim=1)

        return preds                # [N H W] with {0..C-1}

    def start(self):
        # Load the net
        checkpoint = torch.load(self.cfg.testing.checkpoint_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        # Test the net
        with torch.no_grad():
            for i, images in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                preds = self.forward(images)
                display(images, preds,
                        save=self.cfg.testing.save_predictions,
                        save_dir=self.cfg.testing.testing_dir,
                        fname=f"pred_{i}.eps")


def display(images, preds, save=False, save_dir=None, fname=None):
    fig, ax = plt.subplots(2, 1)
    disp_pred = preds[0].cpu().numpy()
    disp_image = images[0].permute(1, 2, 0).cpu().numpy()

    ax[0].imshow(disp_pred)
    ax[1].imshow(disp_image)

    ax[0].set_title("Predictions")
    ax[1].set_title("Images")

    if save:
        save_path = Path(save_dir) / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="eps", dpi=1200)
    else:
        plt.show()
