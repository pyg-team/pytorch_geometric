import zipfile


def extract_zip(path, folder):
    print('Extracting', path)

    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)
