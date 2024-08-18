"""
Demo of SiamCAR adapted on 360° videos
"""
import os
from tracker360.track import TrackerHead360
cwd = os.getcwd()

video_dir = "manipulation_data"
model_dir = "SavedModels"
# video_file = "YassirDocRoom.mp4"
video_file = "2019 04 28 vr360m dog walk ford lake park.mp4"

model_file = "MinLossModel.pt"

video_path = os.path.join(os.path.join(cwd, video_dir), video_file)
model_path = os.path.join(os.path.join(cwd, model_dir), model_file)

tracker_head = TrackerHead360(model_path=model_path)
tracker_head.track_360(video_path, start_idx=3703, end_idx=5200, adapt_fov=True)

# tracker_head.track_2d(video_path)

