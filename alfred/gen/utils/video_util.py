import os
import subprocess
import alfred.gen.constants as constants

class VideoSaver(object):

    def __init__(self, frame_rate=constants.VIDEO_FRAME_RATE):
        self.frame_rate = frame_rate

    def save(self, image_path, save_path):
        with open(os.devnull, 'w') as devnull:
            subprocess.call(["ffmpeg -r %d -pattern_type glob -y -i '%s' -c:v libx264 -pix_fmt yuv420p '%s'" %
                             (self.frame_rate, image_path, save_path)], shell=True, stdout=devnull, stderr=devnull)
