from huggingface_hub import snapshot_download
import os

repo_id = "openvla/openvla-7b-finetuned-libero-goal"

save_directory = "./SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"

if __name__ == "__main__":
    print(f"Downloading model: '{repo_id}'")
    print(f"Saving to: '{os.path.abspath(save_directory)}'")

    snapshot_download(
        repo_id=repo_id,
        local_dir=save_directory,
        local_dir_use_symlinks=False,  # Set to False to copy files instead of symlinking (more portable)
        resume_download=True           # Resume a download if it was interrupted
    )

    print("\n✅ Download complete!")
    print(f"Model files are now in the '{save_directory}' directory.")
