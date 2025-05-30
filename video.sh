fmpeg -i output_segmented.mp4 -vcodec libx264 -acodec aac -vf scale=1280:720 -r 30 -b:v 1500k -b:a 128k output_video.mp4

