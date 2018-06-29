import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed, dilation
from skimage.feature import peak_local_max
from skimage.io import imread
from scipy import ndimage
from skimage import img_as_float
import cv2


# Generate an initial image with two overlapping circles
image = img_as_float(imread("/home/qamaruddin/PycharmProjects/ariel/RAW_data/002/IMG_1210.JPG"))

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
# Black tophat transformation (see https://en.wikipedia.org/wiki/Top-hat_transform)
hat = ndimage.black_tophat(image, 7)
# Combine with denoised image
hat -= 0.3 * image
# Morphological dilation to try to remove some holes in hat image
hat = dilation(hat)
markers = np.zeros_like(image)
markers[10, 10, :] = 1
markers[1700, 1500, :] = 1
labels = watershed(hat, markers, mask=image)

image = cv2.imread("/home/qamaruddin/PycharmProjects/ariel/RAW_data/002/IMG_1210.JPG")
# algorithm
for label in np.unique(labels):
    # if the label is zero, we are examining the 'background'
    # so simply ignore it
    if label == 0:
        continue

    # otherwise, allocate memory for the label region and draw
    # it on the mask
    mask = np.zeros(image.shape, dtype="uint8")
    mask[labels == label] = 255

    # detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)

    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
    cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)


segs = np.where(labels > 0, image, 0)
print(np.where(labels > 0))
# np.logical_and(image, labels).astype(np.uint8)
fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
# ax[1].imshow(hat, cmap=plt.cm.gray, interpolation='nearest')
# ax[1].set_title('Distances')
ax[1].imshow(segs, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Segmented objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()