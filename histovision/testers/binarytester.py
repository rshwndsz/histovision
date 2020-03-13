import torch
import matplotlib.pyplot as plt
import hydra
from tqdm import tqdm
from pathlib import Path
import cv2

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
                        fname=f"pred_{i}.png")


def display(images, preds, save=False, save_dir=None, fname=None):
    if not save:
        # Take one (image, pred) pair from the batch
        disp_pred = preds[0].cpu().numpy()
        disp_image = images[0].permute(1, 2, 0).cpu().numpy()

        fig, ax = plt.subplots(2, 1)
        fig.tight_layout(pad=4.0)
        plt.axis('off')
        ax[0].imshow(disp_pred, "gray")
        ax[1].imshow(disp_image)
        ax[0].set_title("Prediction")
        ax[1].set_title("Image")
        # Display both image and prediction in one plot
        plt.show()
    else:
        save_path = Path(save_dir) / fname
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # If predictions are in a batch, save each one separately
        if len(preds.size()) > 3:
            for pred in preds:
                print(pred.size(), preds.size())
                cv2.imwrite(save_path, pred.permute(1, 2, 0).cpu().numpy())
        # Else save the one prediction
        else:
            plt.axis('off')
            plt.imshow(preds.permute(1, 2, 0).cpu().squeeze().numpy(), cmap="gray")
            plt.savefig(save_path, bbox_inches='tight')
