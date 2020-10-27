import os
import mimetypes
from pathlib import Path
from functools import partial

get_ext = lambda x: [i for i,j in mimetypes.types_map.items() if j.startswith(x)]

image_ext = get_ext('image')
video_ext = get_ext('video')
text_ext = get_ext('video')
audio_ext = get_ext('audio')

join_path = lambda x,y: Path(os.path.join(x, y))

def get_files(root, file_type=None, recursive=True):
    files = []
    if file_type: file_type = tuple(file_type)
    if not recursive:
        files.extend([join_path(root, i) for i in os.listdir(root) if str(i).lower().endswith(file_type)])
    else:
        for p, d, fs in os.walk(root):
            if file_type:
                files.extend([join_path(p, f) for f in fs if str(f).lower().endswith(file_type)])
            else:
                files.extend([join_path(p, f) for f in fs])
    return files

get_image_files = partial(get_files, file_type = image_ext)
get_video_files = partial(get_files, file_type = video_ext)
get_text_files = partial(get_files, file_type = text_ext)
get_audio_files = partial(get_files, file_type = audio_ext)
