import os
import csv

def get_video_paths(main_folder):
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    video_paths = {}
    for idx, subfolder in enumerate(subfolders):
        videos = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
        video_paths[idx] = videos
    return video_paths, subfolders

def write_csvs(video_paths, subfolders, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    first_50_csv = os.path.join(output_folder, 'first_50_videos.csv')
    remainder_csv = os.path.join(output_folder, 'remainder_videos.csv')
    codex_csv = os.path.join(output_folder, 'codex.csv')

    with open(first_50_csv, 'w', newline='') as f50, open(remainder_csv, 'w', newline='') as remainder:
        f50_writer = csv.writer(f50)
        remainder_writer = csv.writer(remainder)

        for idx, paths in video_paths.items():
            for i, path in enumerate(paths):
                if i < 200:
                    f50_writer.writerow([path, idx])
                else:
                    remainder_writer.writerow([path, idx])
    
    with open(codex_csv, 'w', newline='') as codex:
        codex_writer = csv.writer(codex)
        for idx, subfolder in enumerate(subfolders):
            codex_writer.writerow([idx, os.path.basename(subfolder)])

# Define your main folder and output folder
main_folder = "/home/leon-gold/Downloads/TrialMoonwalk/New Sim Classes"
output_folder = '/home/leon-gold/jepa/csvs/0NewSim'

# Get video paths and subfolders
video_paths, subfolders = get_video_paths(main_folder)

# Write CSVs
write_csvs(video_paths, subfolders, output_folder)

print("CSV files created successfully!")
