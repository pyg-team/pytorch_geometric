import argparse
import logging
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import time
from functools import partial
from threading import Thread
from typing import Optional


def clean_runs(get_all_remote_pids, conn):
    """This process cleans up the remaining remote training tasks."""
    print("Cleanup runs")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    data = conn.recv()

    # If the launch process exits normally, don't do anything:
    if data == "exit":
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        for (ip, port), pids in remote_pids.items():
            kill_proc(ip, port, pids)
    print("Cleanup exits")


def kill_proc(ip, port, pids):
    """SSH to remote nodes and kill the specified processes."""
    curr_pid = os.getpid()
    killed_pids = []
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print(f"Kill process {pid} on {ip}:{port}", flush=True)
        kill_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) + " " +
                    ip + " 'kill {}'".format(pid))
        subprocess.run(kill_cmd, shell=True)
        killed_pids.append(pid)
    for i in range(3):
        killed_pids = get_pids_to_kill(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break
        else:
            killed_pids.sort()
            for pid in killed_pids:
                print(f"Kill process {pid} on {ip}:{port}", flush=True)
                kill_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) +
                            " " + ip + " 'kill -9 {}'".format(pid))
                subprocess.run(kill_cmd, shell=True)


def get_pids_to_kill(ip, port, killed_pids):
    """Get the process IDs that we want to kill but are still alive."""
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ",".join(killed_pids)
    ps_cmd = ("ssh -o StrictHostKeyChecking=no -p " + str(port) + " " + ip +
              " 'ps -p {} -h'".format(killed_pids))
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode("utf-8").split("\n"):
        ps = p.split()
        if len(ps) > 0:
            pids.append(int(ps[0]))
    return pids


def remote_execute(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
    port: int,
    username: Optional[str] = None,
) -> Thread:
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        state_q: A queue collecting Thread exit states.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        username: If given, this will specify a username to use when issuing
            commands over SSH. Useful when your infra requires you to
            explicitly specify a username to avoid permission issues.

    Returns:
        thread: The thread who runs the command on the remote host.
            Returns when the command completes on the remote host.
    """
    ip_prefix = ""
    if username is not None:
        ip_prefix += f"{username}@"

    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} "
               f"'{cmd}'")

    print(f"----- ssh_cmd={ssh_cmd} ")

    # thread func to run the job
    def run(ssh_cmd, state_q):
        try:
            subprocess.check_call(ssh_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception:
            state_q.put(-1)

    thread = Thread(
        target=run,
        args=(
            ssh_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # Sleep for a while in case SSH is rejected by peer due to busy connection:
    time.sleep(0.2)
    return thread


def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine."""
    pids = []
    curr_pid = os.getpid()
    # We want to get the Python processes. However, we may get some SSH
    # processes, so we should filter them out:
    ps_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip} "
              f"'ps -aux | grep python | grep -v StrictHostKeyChecking'")
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode("utf-8").split("\n"):
        ps = p.split()
        if len(ps) < 2:
            continue
        # We only get the processes that run the specified command:
        res = re.search(cmd_regex, p)
        if res is not None and int(ps[1]) != curr_pid:
            pids.append(ps[1])

    pid_str = ",".join([str(pid) for pid in pids])
    ps_cmd = (f"ssh -o StrictHostKeyChecking=no -p {port} {ip} "
              f" 'pgrep -P {pid_str}'")
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode("utf-8").split("\n")
    all_pids = []
    for pid in set(pids + pids1):
        if pid == "" or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids


def get_all_remote_pids(hosts, ssh_port, udf_command):
    """Get all remote processes."""
    remote_pids = {}
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # When creating training processes in remote machines, we may insert
        # some arguments in the commands. We need to use regular expressions to
        # match the modified command.
        cmds = udf_command.split()
        new_udf_command = " .*".join(cmds)
        pids = get_remote_pids(ip, ssh_port, new_udf_command)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids


def wrap_cmd_w_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired environment variables.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_w_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"
    """
    if env_vars == "":
        return f"({cmd})"
    else:
        return f"(export {env_vars}; {cmd})"


def wrap_cmd_w_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra environment variables.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_w_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_w_envvars(cmd, env_vars)


def get_available_port(ip):
    """Get available port with specified ip."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(1234, 65535):
        try:
            sock.connect((ip, port))
        except Exception:
            return port
    raise RuntimeError(f"Failed to get available port for ip~{ip}")


def submit_all_jobs(args, udf_command, dry_run=False):
    if dry_run:
        print("Dry run mode, no jobs will be launched")

    servers_cmd = []
    hosts = []
    thread_list = []

    # Get the IP addresses of the cluster:
    ip_config = os.path.join(args.workspace, args.ip_config)
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of 'ip_config'")

    state_q = queue.Queue()

    master_ip, _ = hosts[0]
    for i in range(len(hosts)):
        ip, _ = hosts[i]
        server_env_vars_cur = ""
        cmd = wrap_cmd_w_envvars(udf_command, server_env_vars_cur)
        cmd = (wrap_cmd_w_extra_envvars(cmd, args.extra_envs)
               if len(args.extra_envs) > 0 else cmd)

        cmd = cmd[:-1]
        cmd += f" --dataset_root_dir={args.dataset_root_dir}"
        cmd += f" --dataset={args.dataset}"
        cmd += f" --num_nodes={args.num_nodes}"
        cmd += f" --num_neighbors={args.num_neighbors}"
        cmd += f" --node_rank={i}"
        cmd += f" --master_addr={master_ip}"
        cmd += f" --num_epochs={args.num_epochs}"
        cmd += f" --batch_size={args.batch_size}"
        cmd += f" --num_workers={args.num_workers}"
        cmd += f" --concurrency={args.concurrency}"
        cmd += f" --logging --progress_bar --ddp_port={args.ddp_port})"
        servers_cmd.append(cmd)

        if not dry_run:
            thread_list.append(
                remote_execute(cmd, state_q, ip, args.ssh_port,
                               username=args.ssh_username))

    # Start a cleanup process dedicated for cleaning up remote training jobs:
    conn1, conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, udf_command)
    process = multiprocessing.Process(target=clean_runs, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info("Stop launcher")
        # We need to tell the cleanup process to kill remote training jobs:
        conn2.send("cleanup")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            err = err_code  # Record error code:

    # The training processes completed.
    # We tell the cleanup process to exit.
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)
    print("=== fully done ! === ")


def main():
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument(
        "--ssh_port",
        type=int,
        default=22,
        help="SSH port",
    )
    parser.add_argument(
        "--ssh_username",
        type=str,
        default="",
        help=("When issuing commands (via ssh) to the cluster, use the "
              "provided username in the ssh cmd. For example, if you provide "
              "--ssh_username=bob, then the ssh command will be like "
              "'ssh bob@1.2.3.4 CMD'"),
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path of user directory of distributed tasks",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="The name of the dataset",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default='../../data/products',
        help="The root directory (relative path) of partitioned dataset",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=2,
        help="Number of distributed nodes",
    )
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="15,10,5",
        help="Number of node neighbors sampled at each layer",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The current node rank",
    )
    parser.add_argument(
        "--num_training_procs",
        type=int,
        default=2,
        help="The number of training processes per node",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default='localhost',
        help="The master address for RPC initialization",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="The number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training and testing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of sampler sub-processes",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of maximum concurrent RPC for each sampler",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0004,
        help="Learning rate",
    )
    parser.add_argument(
        '--ddp_port',
        type=int,
        default=11111,
        help="Port used for PyTorch's DDP communication",
    )
    parser.add_argument(
        "--part_config",
        type=str,
        required=True,
        help="File (in workspace) of the partition configuration",
    )
    parser.add_argument(
        "--ip_config",
        required=True,
        type=str,
        help="File (in workspace) of IP configuration for server processes",
    )
    parser.add_argument(
        "--extra_envs",
        nargs="+",
        type=str,
        default=[],
        help=("Extra environment parameters be set. For example, you can set "
              "the 'LD_LIBRARY_PATH' by adding: --extra_envs "
              "LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"),
    )
    args, udf_command = parser.parse_known_args()

    udf_command = str(udf_command[0])
    if "python" not in udf_command:
        raise RuntimeError("Launching script does only support a Python "
                           "executable file")
    submit_all_jobs(args, udf_command)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
