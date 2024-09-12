from pytube import YouTube 
import urllib, urllib3

# where to save 
SAVE_PATH = "src/datasets/Apollo17" #to_do 

# link of the video to be downloaded 
link = "https://www.youtube.com/watch?v=GakAd6epHko"


try: 
    # object creation using YouTube 
    yt = YouTube(link) 
except: 
    #to handle exception 
    print("Connection Error") 

# Get all streams and filter for mp4 files
mp4_streams = yt.streams.filter(file_extension='mp4').all()

# get the video with the highest resolution
d_video = mp4_streams[-1]

try: 
    # downloading the video 
    d_video.download(output_path=SAVE_PATH)
    print('Video downloaded successfully!')
except: 
    print("Some Error!")

