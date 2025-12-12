from huggingface_hub import snapshot_download

# Define the repository ID
repo_id = "openvla/modified_libero_rlds"

# Define the local directory where you want to save the data
# Based on your previous errors, you should save it where your script expects it:
local_dir = "/scratch/users/jjosh/spec/SpecVLA/dataset/modified_libero_rlds"

print(f"Starting download of {repo_id} to {local_dir}...")

# Download the entire repository snapshot
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    # This ensures that if you stop and restart, it continues from where it left off
    local_dir_use_symlinks=False, 
    # Optional: If you only want the specific commit/version
    # revision="main"
)

print("Download complete!")

