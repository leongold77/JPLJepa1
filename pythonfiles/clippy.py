from moviepy.editor import VideoFileClip
import os
import csv

def time_to_seconds(time_str):
    """Convert time in 'HH:MM:SS' format to seconds."""
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def extract_clip(input_video_path, start_time, end_time, output_path):
    """Extract a clip from the video and save it."""
    video = VideoFileClip(input_video_path)
    start_seconds = time_to_seconds(start_time)
    end_seconds = time_to_seconds(end_time)
    clip = video.subclip(start_seconds, end_seconds)
    clip.write_videofile(output_path, codec="libx264", audio=False)

def time_to_seconds_to_str(seconds):
    """Convert seconds to 'HH:MM:SS' format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# Set path to the local video file
local_video_path = r"Greatest long jump final ever： Carl Lewis and Mike Powell trade world records in 1991 ｜ NBC Sports [sLmoJyVnLm0].webm"  # Update this path to your local video file

# Define clips to extract with labels
clips =[{"start": "00:00:31", "end": "00:00:33", "label": "Jumping2"},
{"start": "00:02:13", "end": "00:02:16", "label": "Jumping2"},
{"start": "00:03:37", "end": "00:03:39", "label": "Jumping2"},
{"start": "00:04:09", "end": "00:04:12", "label": "Jumping2"},
{"start": "00:04:58", "end": "00:05:00", "label": "Jumping2"}
]






# Create a base output directory
base_output_directory = "/home/leon-gold/Downloads/TrialMoonwalk/Earth"
os.makedirs(base_output_directory, exist_ok=True)

# Create and write to metadata CSV file
metadata_file = os.path.join(base_output_directory, "metadata.csv")
with open(metadata_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Start Time", "End Time", "Label"])

    for i, clip in enumerate(clips):
        label_folder = os.path.join(base_output_directory, clip["label"])
        os.makedirs(label_folder, exist_ok=True)
       
        # Extract the specified clip
        output_path = os.path.join(label_folder, f"clip_{i:04d}.mp4")
        extract_clip(local_video_path, clip["start"], clip["end"], output_path)
       
        # Write metadata for each clip
        writer.writerow([f"clip_{i:04d}.mp4", clip["start"], clip["end"], clip["label"]])

print(f"Clips have been extracted and saved in labeled folders. Metadata saved in '{metadata_file}'")

