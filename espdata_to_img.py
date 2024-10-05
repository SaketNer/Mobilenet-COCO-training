import numpy as np
from PIL import Image

# Assuming `data` is your 240x240x3 numpy array with RGB888 format
data = [  ]


flat_array = np.array(data, dtype=np.uint8)
print(flat_array.size)

if flat_array.size != 172800:
    raise ValueError("Input data must contain exactly 1,728,000 values.")


img_array = flat_array.reshape((240, 240, 3))


img = Image.fromarray(img_array, 'RGB')


img.save('./temp/test_image2.png')  # Save the image
