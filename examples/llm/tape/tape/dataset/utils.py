import hashlib


def generate_string_hash(input_string: str, algorithm: str = 'sha256'):
    input_bytes = input_string.encode('utf-8')
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(input_bytes)
    return hash_obj.hexdigest()
