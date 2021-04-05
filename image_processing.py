import cv2
import numpy as np
import tensorflow as tf

def centerWithinRect(xpos, ypos, width, height, centerpos):
    centerx, centery = centerpos
    return centerx in range(xpos, xpos+width) and centery in range(ypos, ypos+height)

def resize_image(img, size=(64,64)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*", "/"]
original = cv2.imread("testing-pics/5.PNG")
workingPhoto = np.copy(original)

image = cv2.blur(original, (3, 3))
image = cv2.erode(image, np.ones((3, 3),np.uint8))
image = cv2.dilate(image, np.ones((3, 3),np.uint8))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 4)
cv2.imshow("post_canny", image)
cv2.waitKey(0)
image = cv2.dilate(image, np.ones((3, 3),np.uint8))


cv2.imshow("post_canny", image)
cv2.waitKey(0)

queue = []
model = tf.keras.models.load_model("test.model")
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    x, y, width, height = cv2.boundingRect(contour)
    cv2.rectangle(workingPhoto, (x,y), (x + width, y + height), (255, 0, 0), 2)

    snip = original[y:y+height,x:x+width]
    gray = cv2.cvtColor(snip, cv2.COLOR_BGR2GRAY)

    _, snip_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    snip_thresh = cv2.copyMakeBorder(snip_thresh, 9, 9, 9, 9, cv2.BORDER_CONSTANT, 0)

    result = resize_image(snip_thresh, (64,64))

    cv2.imshow("result", result)
    cv2.waitKey(0)
    queue.append((x, result))

queue.sort(key = lambda x : x[0])
queue = [result for x, result in queue]
queue = np.array(queue).reshape(-1, 64,64, 1) / 255

cv2.imshow("bounding boxes", workingPhoto)

label_indexes = list(map(np.argmax, model.predict(queue)))
prediction = [labels[index] for index in label_indexes]
print(prediction)
print("".join(prediction))
print(eval("".join(prediction)))

cv2.waitKey(0)
cv2.destroyAllWindows()
