from PIL import Image
import os
import sys


directory = r'C:\Users\User\Desktop\ALLSTIMULI'

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))

  new_dimensions = (300,300)
  output = image.resize(new_dimensions, Image.ANTIALIAS)
  print(output)

  output_file_name = os.path.join(directory, "small_" + file_name)
  output.save(output_file_name, "JPEG", quality = 95)

print("All done")