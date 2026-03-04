import cv2 as cv
import numpy as np
import urllib.request

def read_image_url(url):
    req = urllib.request.urlopen(url)   # open the URL
    img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(img_rw, cv.IMREAD_GRAYSCALE)  # decode as grayscale image
    return img
def add_muoi_tieu(img, prob):
    mean = 0
    sigma = 50
    noisy = np.random.normal(mean, sigma, img.shape)
    new_img = np.clip(img + noisy, 0, 255).astype(np.uint8)
    return new_img

if __name__=="__main__":
    url="https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/solidWhiteCurve.jpg"
    anh_goc=read_image_url(url)
    anh_muoi_tieu= add_muoi_tieu(anh_goc, 0.03)
    img2= anh_muoi_tieu.copy()
    clean_img = cv.blur(img2, (3,3))
    img3 = np.concatenate((anh_muoi_tieu, clean_img), axis=1)
    cv.imshow("img3", img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img5 = anh_muoi_tieu.copy()
    clean_img = cv.medianBlur(img5, 3)
    im6= np.concatenate((anh_muoi_tieu, clean_img), axis=1)
    cv.imshow("img3", img6)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ed1= cv.Canny(anh_muoi_tieu, 50, 150)
    ed2= cv.Canny(clean_img, 50, 150)
    ed3= cv.Canny(anh_goc, 50, 150)
    img7= np.concatenate((ed1, ed2, ed3), axis=1)
    cv.imshow("img7", img7)
    cv.waitKey(0)
    cv.destroyAllWindows()
    