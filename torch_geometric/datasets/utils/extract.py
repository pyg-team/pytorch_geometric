import tarfile
import zipfile
import subprocess

from .spinner import Spinner


def extract_tar(file_path, dir, mode='r:gz'):
    spinner = Spinner('Extracting', file_path).start()

    with tarfile.open(file_path, mode) as tar:
        tar.extractall(dir)
        spinner.success()


def extract_zip(file_path, dir):
    spinner = Spinner('Extracting', file_path).start()

    with zipfile.ZipFile(file_path, 'r') as zip:
        zip.extractall(dir)
        spinner.success()


def extract_7z(file_path, dir):
    spinner = Spinner('Extracting', file_path).start()

    cmd = ['7z', '-y', 'e', file_path, '-o{}'.format(dir)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    spinner.success()
