import tarfile
import zipfile


def extract_tar(path, folder, mode='r:gz'):
    print('Extracting', path)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path, folder):
    print('Extracting', path)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
