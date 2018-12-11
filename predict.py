from utils import *
import matplotlib.pyplot as plt
seg = fingernailseg()
print('create_unet')
seg.create_unet()
seg.load_model()
print('predicting')

mask = seg.predict()
raw = seg.X_test
for i in range(0,seg.X_test.__len__()):
  plt.figure(figsize=(5,5))
  rand_image = i
  plt.imshow(raw[rand_image,:,:,:])
  plt.imshow(mask[rand_image,:,:,0], alpha=0.8)
  #plt.title('Fingernails segmentation of test image', fontsize=15)
  filename = str(rand_image)+'.png'
  plt.savefig(filename)
  print('saved',filename)
print('done')
