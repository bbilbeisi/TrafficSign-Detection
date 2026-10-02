import os

MAX_DEPTH = 3  # Depth level to traverse
EXCLUDE_DIRS = {'.git', '__pycache__', 'venv', '.venv', 'env'}

def get_dir_summary(path, current_depth=0):
    if current_depth > MAX_DEPTH:
        return
    
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        return

    for entry in entries:
        if entry in EXCLUDE_DIRS:
            continue
        
        full_path = os.path.join(path, entry)
        indent = "│   " * current_depth + "├── "
        
        if os.path.isdir(full_path):
            print(f"{indent}{entry}/")
            get_dir_summary(full_path, current_depth + 1)
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            size_str = f"({size_mb:.2f} MB)" if size_mb >= 1.0 else ""
            print(f"{indent}{entry} {size_str}")

if __name__ == "__main__":
    print(f"Directory breakdown for: {os.getcwd()}\n")
    get_dir_summary(".")