# === Patch deprecated NumPy aliases in X-CLIP dataloaders ===
import os, re, pathlib

repo_root = "/home/jjhong/X-CLIP"
print("Patching float ...")

for path in pathlib.Path(repo_root).rglob("*.py"):
    text = path.read_text()
    new_text = re.sub(r"\bnp\.float\b", "float", text)
    if new_text != text:
        path.write_text(new_text)
        print(f"Patched: {path}")

print("✅ Done patching.")