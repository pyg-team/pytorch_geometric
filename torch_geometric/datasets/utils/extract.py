import tarfile


def extract_tar(file_path, dir, mode='r:gz'):
    with tarfile.open(file_path, mode) as tar:
        tar.extractall(dir)
