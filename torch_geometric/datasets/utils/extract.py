import tarfile
import zipfile


def extract_tar(file_path, dir, mode='r:gz'):
    with tarfile.open(file_path, mode) as tar:
        tar.extractall(dir)


def extract_zip(file_path, dir):
    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(dir)
