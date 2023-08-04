import argparse

import click
import paramiko
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='dist_train_sage_sup_config.yml',
        help='The path to the configuration file',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='The number of training epochs',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='The batch size for the training and testing data loaders',
    )
    parser.add_argument(
        '--master_addr',
        type=str,
        default='0.0.0.0',
        help='Master IP address for synchronization across all training nodes',
    )
    parser.add_argument(
        '--master_port',
        type=str,
        default='11345',
        help='The port for synchronization across all training nodes',
    )
    args = parser.parse_args()

    config = open(args.config, 'r')
    config = yaml.safe_load(config)
    dataset = config['dataset']
    ip_list = config['nodes']
    port_list = config['ports']
    username_list = config['usernames']
    dst_path_list = config['dst_paths']
    node_ranks = list(range(len(ip_list)))
    num_nodes = len(node_ranks)
    visible_devices = config['visible_devices']
    python_bins = config['python_bins']
    num_cores = len(str(visible_devices[0]).split(','))
    in_channel = str(config['in_channel'])
    out_channel = str(config['out_channel'])

    dataset_path = '../../../data/'
    passwd_dict = {}
    for username, ip in zip(username_list, ip_list):
        passwd_dict[ip + username] = click.prompt(
            f'Password for {username}@{ip}', hide_input=True)
    for username, ip, port, dst, noderk, device, pythonbin in zip(
            username_list,
            ip_list,
            port_list,
            dst_path_list,
            node_ranks,
            visible_devices,
            python_bins,
    ):
        trans = paramiko.Transport((ip, port))
        trans.connect(username=username, password=passwd_dict[ip + username])
        ssh = paramiko.SSHClient()
        ssh._transport = trans

        to_dist_dir = 'cd ' + dst + \
            '/examples/distributed/graphlearn_for_pytorch/ '
        exec_example = "tmux new -d 'CUDA_VISIBLE_DEVICES=" + str(device) + \
            " " + pythonbin + " dist_train_sage_supervised.py --dataset=" + \
            dataset + " --dataset_root_dir=" + dataset_path + dataset + \
            " --in_channel=" + in_channel + " --out_channel=" + out_channel + \
            " --node_rank=" + str(noderk) + " --num_dataset_partitions=" + \
            str(num_nodes) + " --num_nodes=" + str(num_nodes) + \
            " --num_training_procs=" + str(num_cores) + " --master_addr=" + \
            args.master_addr + " --training_pg_master_port=" + \
            args.master_port + " --train_loader_master_port=" + \
            str(int(args.master_port) + 1) + " --test_loader_master_port=" + \
            str(int(args.master_port) + 2) + " --batch_size=" + \
            str(args.batch_size) + " --epochs=" + str(args.epochs)

        print(to_dist_dir + ' && ' + exec_example + " '")
        stdin, stdout, stderr = ssh.exec_command(
            to_dist_dir + ' && ' + exec_example + " '", bufsize=1)
        print(stdout.read().decode())
        print(stderr.read().decode())
        ssh.close()
