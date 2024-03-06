import os 

def is_file_path(path):
    return os.path.exists(path)

def is_http_path(path):
    return path.startswith("http://") or path.startswith("https://")

