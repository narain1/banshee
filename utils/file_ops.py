from pathlib import Path
from typing import Union, String

def read_file(x: Union[String, Path]):
    if isinstance(x, str): x = Path(x)
    return x.read_file()

def read_rgb(x):
    img = cv2.imread(x, cv2.IMREAD_COLOR)
    img = cv.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

