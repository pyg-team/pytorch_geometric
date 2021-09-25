import os
from os.path import join
import argparse
import torch
from torch_geometric.datasets.mesh_cnn_dataloader import MeshDataLoader
from torch_geometric.datasets.mesh_cnn_datasets import MeshShrech16Dataset, \
    MeshCubesDataset
from torch_geometric.nn.models.mesh_models import define_classifier, \
    define_loss, get_scheduler
import time


class MeshCNNClassification:
    r"""This class is an example of MeshCNN use for classification task as
    described in the paper: `"MeshCNN: A Network with an Edge"
    <https://arxiv.org/abs/1809.05910>`_ paper
    Args:
        gpu_ids (int, list): GPU IDs to use in the dataloader.
        train (bool): `True` if this is a trainig case, `False` otherwise.
        checkpoints_dir (str): Path to save intermediate network files.
        name (str): Network name.
        nclasses (int): number of input classes.
        input_nc (int): number of channels in the edge features data.
        ncf (list of ints): number of convolution filters list.
        ninput_edges (int): number of input edges.
        continue_train (bool, optional): If `True` - will continue the training
                                         from a specific epoch ('which_epoch')
                                         in 'checkpoints_dir'. Default is
                                         `False` - which means train from
                                         scratch.
        which_epoch (str, optional): If continue_train set to `True` - it will
                                     use saved network weights from this epoch.
                                     Default is 'latest'.
        export_folder (str, optional): an export folder to create intermediate
                                       results for visualization. Default is an
                                       empty path which means no data export.
    """

    def __init__(self, gpu_ids, train, checkpoints_dir, name, nclasses,
                 input_nc, ncf, ninput_edges,
                 continue_train=False, which_epoch='latest', export_folder=''):
        super(MeshCNNClassification, self).__init__()
        self.gpu_ids = gpu_ids
        self.train = train
        self.continue_train = continue_train
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(checkpoints_dir, name)

        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.export_folder = export_folder
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        self.edge_index = None

        self.nclasses = nclasses

        # load/define network
        self.net = define_classifier(input_nc, ncf, ninput_edges, nclasses,
                                     norm_type='group', num_groups=16,
                                     resblocks=1,
                                     pool_res=[600, 450, 300, 180],
                                     fc_n=100, gpu_ids=gpu_ids,
                                     arch='mconvnet', init_type='normal',
                                     init_gain=0.02)

        self.net.train(self.train)
        self.criterion = define_loss(dataset_mode='classification').to(
            self.device)

        if self.train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0002,
                                              betas=(0.9, 0.999))
            self.scheduler = get_scheduler(self.optimizer, lr_policy='lambda',
                                           epoch_count=1, niter=100,
                                           niter_decay=100)

        if not self.train or self.continue_train:
            self.load_network(which_epoch)

    def forward(self, data):
        self.mesh, self.edge_features, self.labels, _ = data
        out = self.net(self.edge_features, self.mesh)
        return out

    def forward_backward(self, data):
        self.optimizer.zero_grad()
        self.net.train()
        out = self.forward(data)
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def train_model(self, dataloader, epoch, max_epochs):
        epoch_iter = 0
        iter_data_time = time.time()
        epoch_start_time = time.time()
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time
            self.forward_backward(data)
            epoch_iter += dataloader.batch_size
            t = (time.time() - iter_start_time) / dataloader.batch_size
            message = \
                '(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f ' \
                % (epoch, epoch_iter, t, t_data, self.loss.item())
            print(message)
            iter_data_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, max_epochs, time.time() - epoch_start_time))

        self.save_network('latest')
        self.save_network('epoch_{}'.format(epoch))
        self.update_learning_rate()

    def test_model(self, dataloader):
        num_correct_counter = 0
        num_examples_counter = 0
        self.net.eval()
        for i, data in enumerate(dataloader):
            with torch.no_grad():
                out = self.forward(data)
                # compute number of correct
                pred_class = out.data.max(1)[1]
                label_class = self.labels
                correct = pred_class.eq(label_class).sum()
                num_correct_counter += correct
                num_examples_counter += len(label_class)

        accuracy = float(num_correct_counter) / num_examples_counter
        return accuracy

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % which_epoch
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, required=True,
                        help='path to save data')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='dataset name [shrec16|cubes]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='models are saved here')
    parser.add_argument('--num_aug', type=int, default=20,
                        help='# of augmentation files')
    parser.add_argument('--scale_verts', type=bool, default=False,
                        help='non-uniformly scale the mesh e.g., in x, y or z')
    parser.add_argument('--slide_verts', type=float, default=0.2,
                        help='percent vertices which will be shifted along the'
                             ' mesh surface')
    parser.add_argument('--flip_edges', type=float, default=0.2,
                        help='percent of edges to randomly flip')
    parser.add_argument('--n_input_edges', type=int, default=750,
                        help='# of input edges (will include dummy edges)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size')
    parser.add_argument('--data_shuffle', type=bool, default=True,
                        help='if true, takes meshes in order, otherwise takes '
                             'them randomly')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='max number of epochs')
    parser.add_argument('--gpu_ids', type=int, default=[0],
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--export_folder', type=str, default='',
                        help='exports intermediate collapses to this folder')
    return parser


def main():
    # must define: --data_root YOUR_PATH --dataset_name shrec16 or cubes
    parser = make_parser()
    args = parser.parse_args()

    # load train and test datasets:
    if args.dataset_name == 'shrec16':
        train_dataset = MeshShrech16Dataset(root=args.data_root, train=True,
                                            num_aug=args.num_aug,
                                            slide_verts=args.slide_verts,
                                            scale_verts=args.scale_verts,
                                            flip_edges=args.flip_edges)

        test_dataset = MeshShrech16Dataset(root=args.data_root, train=False,
                                           num_aug=1)

    elif args.dataset_name == 'cubes':
        train_dataset = MeshCubesDataset(root=args.data_root, train=True,
                                         num_aug=args.num_aug,
                                         slide_verts=args.slide_verts,
                                         scale_verts=args.scale_verts,
                                         flip_edges=args.flip_edges)

        test_dataset = MeshCubesDataset(root=args.data_root, train=False,
                                        num_aug=1)

    # create mesh dataloader:
    train_data_loader = MeshDataLoader(mesh_dataset=train_dataset,
                                       data_set_type='classification',
                                       train=True,
                                       gpu_ids=args.gpu_ids,
                                       batch_size=args.batch_size,
                                       shuffle=args.data_shuffle,
                                       hold_history=False)
    test_data_loader = MeshDataLoader(mesh_dataset=test_dataset,
                                      data_set_type='classification',
                                      train=False,
                                      gpu_ids=args.gpu_ids,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      hold_history=False)

    model = MeshCNNClassification(gpu_ids=args.gpu_ids, train=True,
                                  checkpoints_dir=args.checkpoints_dir,
                                  name='MeshCNNClassExample',
                                  nclasses=train_data_loader.n_classes,
                                  input_nc=train_data_loader.n_input_channels,
                                  ncf=[64, 128, 256, 256],
                                  ninput_edges=args.n_input_edges)

    # train and test model:
    for epoch in range(1, args.max_epochs + 1):
        model.train_model(train_data_loader, epoch, args.max_epochs)
        test_acc = model.test_model(test_data_loader)
        message = 'epoch: {}, TEST ACC: [{:.5} %]\n'.format(epoch,
                                                            test_acc * 100)
        print(message)


if __name__ == '__main__':
    main()
