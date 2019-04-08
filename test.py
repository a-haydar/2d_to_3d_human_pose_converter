import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data_loader import PoseDataset
from train import Net


class Inferencing():
    def __init__(self, network_path, dataset_path, validation_set):
        self.network_path = network_path
        self.dataset_path = dataset_path
        self.validation_set = validation_set
        self.device =\
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net()
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(self.network_path, map_location='cpu'))
        self.net.train(False)

    def get3Dcoordinates(self, skel_2d):
        """Returns Z(3D) coordinates of a 2D pose"""
        skel_2d = skel_2d.to(self.device)
        z_out = self.net(skel_2d)
        z_out = z_out.detach().cpu().numpy()
        z_out = z_out.reshape(-1)
        return z_out

    def testSample(self):
        """Inferences on 5 random samples."""
        val_idx = np.load(self.validation_set)
        val_sampler = SubsetRandomSampler(val_idx)
        pose_dataset = PoseDataset(self.dataset_path)
        val_loader = DataLoader(dataset=pose_dataset, batch_size=1,\
            sampler=val_sampler)
        for i in range(5):
            data_iter = iter(val_loader)
            skel_2d, skel_z = next(data_iter)

            # inference
            skel_2d = skel_2d.to(self.device)
            z_out = self.net(skel_2d)

            # show
            skel_2d = skel_2d.cpu().numpy()
            skel_2d = skel_2d.reshape((2, -1), order='F')  # [(x,y) x n_joint]
            z_out = z_out.detach().cpu().numpy()
            z_out = z_out.reshape(-1)
            z_gt = skel_z.numpy().reshape(-1)
            self.show_skeletons(skel_2d, z_out, z_gt)

    def show_skeletons(self, skel_2d, z_out, z_gt=None):
        """Show skeleton in 2D and 3D, includes full upper body and headself.

        Keyword Arguments:
        skel_2d - skeleton with x,y coordinates
        z_out - predicted z coordinates (for 3d)
        z_gt - ground truth z coordinates
        """
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        edges = np.array([[1, 0], [0, 2],[2, 3], [3, 4], [0, 5], [5, 6], [6, 7]])

        ax_2d = ax1
        ax_3d = ax2

        # draw 3d
        for edge in edges:
            ax_3d.plot(skel_2d[0, edge], z_out[edge], skel_2d[1, edge], color='r')
            if z_gt is not None:
                ax_3d.plot(skel_2d[0, edge], z_gt[edge], skel_2d[1, edge], color='g')

        ax_3d.set_aspect('equal')
        ax_3d.set_xlabel("x"), ax_3d.set_ylabel("z"), ax_3d.set_zlabel("y")
        ax_3d.set_xlim3d([-2, 2]), ax_3d.set_ylim3d([2, -2]), ax_3d.set_zlim3d([2, -2])
        ax_3d.view_init(elev=10, azim=-45)

        # draw 2d
        for edge in edges:
            ax_2d.plot(skel_2d[0, edge], skel_2d[1, edge], color='r')

        ax_2d.set_aspect('equal')
        ax_2d.set_xlabel("x"), ax_2d.set_ylabel("y")
        ax_2d.set_xlim([-2, 2]), ax_2d.set_ylim([2, -2])

        plt.show()


if __name__ == "__main__":
    network_path = './models/model_0_0227_.pth'
    dataset_path = './data/panoptic_dataset.pickle'
    validation_set = './data/val_idx.npy'
    test_object = Inferencing(network_path, dataset_path, validation_set)
    test_object.testSample()
