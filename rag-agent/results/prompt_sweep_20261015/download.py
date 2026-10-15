import sys, time
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    t = time.time()
    p = snapshot_download(repo, max_workers=4)
    print(repo, p, f"{time.time()-t:.0f}s", flush=True)
