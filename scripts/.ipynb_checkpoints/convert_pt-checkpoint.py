import h5py, torch, glob, os

h5_dir = "h5_paths"
pt_dir = "pt_paths"
os.makedirs(pt_dir, exist_ok=True)

for h5_path in glob.glob(f"{h5_dir}/*.h5"):
    slide_id = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["features"][:])
    torch.save(feats, os.path.join(pt_dir, f"{slide_id}.pt"))
    print(f"Converted {slide_id}")
