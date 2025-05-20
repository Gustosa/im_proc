import cv2 as cv
import numpy as np
import filters

def main():
    # Exemplo de uma soma de matrizes
    imgUtfpr = cv.imread("images/utfpr.png")
    imgGreen = cv.imread("images/green_bg.png")

    summedImages = filters.addImages(imgUtfpr, imgGreen)
    
    cv.imshow("utfpr logo", imgUtfpr)
    cv.imshow("green", imgGreen)
    cv.imshow("sum", summedImages)

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Exemplo de converter BGR para cinza
    src = cv.imread("images/lena.jpg")

    grey = filters.grayscaleBGR(src)
    greyCv = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    cv.imshow("Normal", src)
    cv.imshow("Grayscale", grey)
    cv.imshow("Grayscale CV", greyCv )

    cv.waitKey(0)
    cv.destroyAllWindows()

    # Exemplo de esmaecimento utilizando um kernel gaussiano (remoção de ruído)
    filteredImg = filters.gaussianBlurBGR(src)
    filteredCv = src.copy()
    filteredCv = cv.GaussianBlur(filteredCv, (3,3), 1)
    sharpenedImg = cv.addWeighted(src, 1.5, filteredImg, -0.5, 0)

    cv.imshow("Normal", src)
    cv.imshow("Gaussian Blur", filteredImg)
    cv.imshow("Gaussian Blur CV", filteredCv)
    cv.imshow("Sharpened Image", sharpenedImg)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

main()