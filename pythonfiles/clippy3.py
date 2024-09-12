import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# Function to clip the video into 2-second increments
def clip_video_in_2_second_increments(video_path, start_time, end_time, output_dir):
    with VideoFileClip(video_path) as video:
        duration = video.duration
        
        # Calculate the start and end times in seconds
        start_seconds = start_time[0] * 60 + start_time[1]
        end_seconds = end_time[0] * 60 + end_time[1]
        
        # Ensure end time doesn't exceed video duration
        end_seconds = min(end_seconds, duration)
        
        current_time = start_seconds
        clip_num = 1
        
        while current_time < end_seconds:
            # Calculate the end time of the current clip
            clip_end = min(current_time + 2, end_seconds)
            
            # Create the clip
            clip = video.subclip(current_time, clip_end)
            
            # Save the clip
            base_name = os.path.basename(video_path)
            name, ext = os.path.splitext(base_name)
            output_filename = os.path.join(output_dir, f"{name}_clip{clip_num:03d}{ext}")
            
            # Determine the correct codec based on file extension
            if ext.lower() == '.webm':
                clip.write_videofile(output_filename, codec="libvpx", audio_codec="libvorbis")
            else:
                clip.write_videofile(output_filename, codec="libx264")
            
            # Update current time and clip number
            current_time = clip_end
            clip_num += 1

# Input video path
video_path = 'Lunar Rover Follows its Tracks Home： Apollo 15 1971 - 4K Colorization 60fps [q2sjqmKeM5g].webm'

# Set start and end times in (minutes, seconds)
start_time = (0, 14)
end_time = (2, 22)

# Output directory
output_dir = 'src/datasets/TrialMoonwalk/POV'
os.makedirs(output_dir, exist_ok=True)

# Clip the video
clip_video_in_2_second_increments(video_path, start_time, end_time, output_dir)

print("Clipping completed.")
