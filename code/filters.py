import cv2 as cv
import numpy as np

def grayscaleBGR(src):
    conversion = np.array([0.2989, 0.5870, 0.1140])

    rows, cols = src.shape[:2]
    greyImg = np.zeros((rows, cols), np.uint8)
    
    r, g, b = cv.split(src)
    r = np.multiply(r, conversion[0])
    g = np.multiply(g, conversion[1])
    b = np.multiply(b, conversion[2])

    for i in range(0, rows):
        for j in range(0, cols):
            greyImg[i,j] = r[i,j] + g[i,j] + b[i,j]
    
    return greyImg

def gaussianBlurBGR(src):
    kernel = np.array([
        [0.0778, 0.1233, 0.0778],
        [0.1233, 0.1953, 0.1233],
        [0.0778, 0.1233, 0.0778]
        ])
    
    filteredImg = src.copy()

    rows, cols = src.shape[:2]
    for i in range(1, rows-1, 1):
        for j in range(1, cols-1, 1):
            pixels = filteredImg[i-1:i+2, j-1:j+2]
            anchor = applyGaussian(pixels, kernel)
            filteredImg[i,j] = anchor
    
    return filteredImg
            
def applyGaussian(pixels, kernel):
    anchorValue = 0

    rows, cols = pixels.shape[:2]
    for i in range(0, rows):
        for j in range(0, cols):
            anchorValue += pixels[i,j] * kernel[i,j]
    
    return anchorValue

def main():
    # img_strong_man = cv.imread("images/strong_man.jpg")
    # img_lizard = cv.imread("images/lizard.png")

    # # Recortar e colar partes da imagem
    #     square = img_lizard[100:200, 100:200]
    #     img_lizard[0:100, 125:225] = square

    # # Modificar pixel por pixel conforme condição Q
    #     rows, cols, channels = img_lizard.shape
    #     for i in range(0, rows):
    #         for j in range (0, cols):
    #             variation = np.array([(i+j)%255, i/(j+1), i%(j+1)]).astype(np.uint8)
    #             cv.subtract(img_lizard[i, j], variation, img_lizard[i, j])
        
    #     cv.imshow("The lizard", img_lizard)
    # # Adição pixel a pixel ponderada (efeito de transparência)
    # img_blend = cv.addWeighted(img_lizard, 0.7, img_strong_man, 0.3, 0)

    # cv.imshow("Blend", img_blend)

    cv.waitKey(0)

