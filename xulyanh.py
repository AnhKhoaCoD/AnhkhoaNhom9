import cv2 as cv
import numpy as np
import urllib.request

def read_img_url(url):
    req = urllib.request.urlopen(url)   # open the URL
    img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(img_rw, cv.IMREAD_COLOR)  # decode as color image
    return img

def add_noise(img):
    mean = 0
    sigma = 50
    noisy = np.random.normal(mean, sigma, img.shape)
    new_img = np.clip(img + noisy, 0, 255).astype(np.uint8)
    return new_img

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    img = read_img_url(url)
    cv.imshow("img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img2 = add_noise(img)
    cv.imshow("img2", img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img3 = np.concatenate((img, img2), axis=1)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()