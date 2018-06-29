# https://www.image-map.net
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/qamaruddin/Downloads/rsz_a_0_0_249_0_0.jpg")
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (49,22,179,193)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,45,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
b_channel, g_channel, r_channel = cv2.split(img)
img = cv2.merge((b_channel, g_channel, r_channel, mask2 * 255))

# print(mask2)
# img = np.where(mask2>0, img, 0)  # *mask2[:,:,np.newaxis]
cv2.imwrite("some.png", img)

plt.imshow(img, cmap="BrBG"),plt.colorbar(),plt.show()
