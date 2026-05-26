import os

print("--- Listing /kaggle/input recursively ---")
for root, dirs, files in os.walk("/kaggle/input"):
    # Limit output depth to keep logs readable
    depth = root.replace("/kaggle/input", "").count(os.sep)
    if depth > 3:
        continue
    indent = "  " * depth
    print(f"{indent}[Dir] {os.path.basename(root) or root}")
    for f in files[:10]:  # print up to 10 files per folder
        print(f"{indent}  [File] {f}")
    if len(files) > 10:
        print(f"{indent}  ... and {len(files) - 10} more files")
