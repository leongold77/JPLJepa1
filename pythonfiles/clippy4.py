import os
from moviepy.editor import VideoFileClip

def split_video(input_video_path, output_dir):
    video = VideoFileClip(input_video_path)
    duration = video.duration

    if duration <= 8:
        print(f"Video {input_video_path} is less than or equal to 8 seconds, skipping.")
        return

    num_segments = int(duration // 4)

    for i in range(num_segments):
        start_time = i * 4
        end_time = min((i + 1) * 4, duration)
        segment = video.subclip(start_time, end_time)
        segment_filename = os.path.join(output_dir, f"{os.path.basename(input_video_path).split('.')[0]}_part_{i + 1}.mp4")
        segment.write_videofile(segment_filename, codec="libx264")
    
    if duration % 4 > 0:
        # Handle any remaining segment
        start_time = num_segments * 4
        segment = video.subclip(start_time, duration)
        segment_filename = os.path.join(output_dir, f"{os.path.basename(input_video_path).split('.')[0]}_part_{num_segments + 1}.mp4")
        segment.write_videofile(segment_filename, codec="libx264")

def process_videos_in_folder(input_folder, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_folder):
        if filename.endswith((".mp4", ".avi", ".mov")):  # Add more extensions if needed
            video_path = os.path.join(input_folder, filename)
            split_video(video_path, output_directory)

if __name__ == "__main__":
    input_folder = "/home/leon-gold/Downloads/TrialMoonwalk/Astronaut standing and servicing (servicing tripod)"
    output_directory = "/home/leon-gold/Downloads/TrialMoonwalk/Clipped"
    process_videos_in_folder(input_folder, output_directory)
