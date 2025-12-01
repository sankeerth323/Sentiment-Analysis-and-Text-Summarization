import pytesseract as pyt
from PIL import Image
print(pyt.image_to_string(Image.open('test.png')))
