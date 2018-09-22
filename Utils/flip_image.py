import cv2
import numpy as np
oringin_images = np.load('../../Data/data/train_data/images.npy')
cv2.namedWindow("oringin",0)
cv2.namedWindow("flipped",0)
for i in range(oringin_images.shape[0]):
    image = oringin_images[i]
    cv2.imshow('oringin', image)
    cv2.waitKey(1)
    image = cv2.flip(image,1)
    oringin_images[i] = image
    cv2.imshow('flipped', image)
    cv2.waitKey(1)
    pass
np.save('../../Data/data/train_data/images_flipped.npy',oringin_images)