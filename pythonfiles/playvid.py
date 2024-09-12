import os
import subprocess

def play_video(video_path):
    # Check if VLC is installed
    if subprocess.run(["which", "vlc"], stdout=subprocess.PIPE).returncode != 0:
        print("VLC media player is not installed. Please install it and try again.")
        return

    # Check if the file exists
    if not os.path.isfile(video_path):
        print(f"The file '{video_path}' does not exist.")
        return

    # Command to play the video using VLC
    command = ["vlc", "--play-and-exit", video_path]

    try:
        # Run the VLC command
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while trying to play the video: {e}")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    play_video(video_path)
