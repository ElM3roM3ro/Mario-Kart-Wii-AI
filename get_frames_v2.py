import sys
# add local python 3.8 installation to path, where Pillow is installed
sys.path.append(r"F:\Python\3.12.0\Lib\site-packages")
from PIL import Image

from dolphin import event

def show_screenshot(width: int, height: int, data: bytes):
    print(f"received {width}x{height} image of length {len(data)}")
    image = Image.frombytes('RGBA', (width,height), data, 'raw')
    image.show()

event.on_framedrawn(show_screenshot)