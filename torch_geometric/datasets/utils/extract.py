import tarfile
import zipfile
import subprocess

from .progress import Progress


def extract_tar(file_path, dir, mode='r:gz'):
    with tarfile.open(file_path, mode) as tar:
        tar.extractall(dir)


def extract_zip(file_path, dir):
    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(dir)


def extract_7z(file_path, dir):
    progress = Progress('Extracting', file_path)

    try:
        cmd = ['7z', '-y', 'e', file_path, '-o{}'.format(dir)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        process.wait()
        progress.success()
    except:
        progress.fail()
