import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument("-e", "--epochs", type=int, default=10,
                    help='number of training epochs.')
parser.add_argument("--runs", type=int, default=4, help='number of runs.')
parser.add_argument("--num_layers", type=int, default=3,
                    help='number of layers.')
parser.add_argument("-b", "--batch_size", type=int, default=1024,
                    help='batch size.')
parser.add_argument("--workers", type=int, default=12,
                    help='number of workers.')
parser.add_argument("--neighbors", type=str, default='15,10,5',
                    help='number of neighbors.')
parser.add_argument("--fan_out", type=int, default=10,
                    help='number of fanout.')
parser.add_argument("--log_interval", type=int, default=10,
                    help='number of log interval.')
parser.add_argument("--hidden_channels", type=int, default=256,
                    help='number of hidden channels.')
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--wd", type=float, default=0.00)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument(
    "--use_undirected_graph",
    default=True,
    action='store_true',
    help="Wether or not to use undirected graph",
)
args = parser.parse_args()
print(args.use_undirected_graph)
