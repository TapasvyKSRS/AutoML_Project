import urllib.request
import tarfile
from pathlib import Path

url = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
data_dir = Path("data")
tar_path = data_dir / "fcnet_tabular_benchmarks.tar.gz"
marker_file = data_dir / ".extracted"

# Create the data directory if it doesn't exist
data_dir.mkdir(exist_ok=True)

# Check if we already finished this process previously
if marker_file.exists():
    print("Data already downloaded and extracted. Skipping.")
else:
    # 1. Download
    if not tar_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    
    # 2. Extract
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            # Strip the first component to mimic --strip-components=1
            parts = member.name.split('/')
            if len(parts) > 1:
                member.name = '/'.join(parts[1:])
                target_path = data_dir / member.name
                
                # Skip old files
                if not target_path.exists():
                    tar.extract(member, path=data_dir)
    
    # 3. Clean up the tar file
    print(f"Deleting {tar_path} to save space...")
    tar_path.unlink()
    
    # 4. Leave a marker so we don't redownload next time
    marker_file.touch()
    
    print("Setup complete.")