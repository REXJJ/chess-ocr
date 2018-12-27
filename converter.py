from PIL import Image
import os
start='./'
filelist=os.listdir(start)
cou=0
for fichier in filelist[:]:
    if (fichier.endswith(".png")):
      print fichier
      if fichier=='test2.png':
         cou=cou+1
         im = Image.open(start+fichier)
         rgb_im = im.convert('RGB')
         rgb_im.save(start+'imag'+str(cou)+'.jpg')

