import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
import os
import sys
import morphology
# 图片重命名
def reNamePicture(picturePath):
    dirs = os.listdir(picturePath)
    pictureId = 1
    for file in dirs:
        os.rename(picturePath + "\\" + file,picturePath + "\\" +str(pictureId) + ".JPG")
        pictureId = pictureId + 1
    sys.stdin.flush()
    print("文件重命名成功")

#图片裁剪
def clipPicture(originalPicturePath,clipPicturePath):
    dirs = os.listdir(originalPicturePath)
    for image in dirs:
        print (image)
        os.chdir(originalPicturePath)
        img = PIL.Image.open(originalPicturePath +"\\" + image)
        print(img.size)
        size = img.size
        length = 200
        width = 200
        print (size[1] / 2 - width,size[0] / 2 - length, size[1] / 2 + width,size[0] / 2 + length)
        cropped = img.crop((size[0] / 2 - width,size[1] / 2 - length, size[0] / 2 + width,size[1] / 2 + length))
        cropped.save(clipPicturePath + "\\" + image)

#图片向量化
def pictureToVector(clipPicturePath,vectorPicturePath):
    dirs = os.listdir(clipPicturePath)
    for image in dirs:
        print (image)

        os.chdir(clipPicturePath)




#轮廓提取
def getOutline(originalPicturePath,outlinePicturePath):
    dirs = os.listdir(originalPicturePath)
    for image in dirs:
        print("从originalPicture输出" +image + "到outlinePicture")
        print(os.path.splitext(image)[0])
        os.chdir(originalPicturePath)
        img = cv2.imread(originalPicturePath + "\\" + image)

        #gaussianBlur = cv2.GaussianBlur(img, (3, 3), 0)
        bilateralFilter = cv2.bilateralFilter(img,7,75,75)
        #canny1 = cv2.Canny(gaussianBlur, 75, 75)
        canny2 = cv2.Canny(bilateralFilter, 30, 150)
        #cv2.imshow('Canny1', canny1)
        cv2.namedWindow(image, cv2.WINDOW_NORMAL)
        cv2.imshow(image, canny2)
        cv2.waitKey(0)
        os.chdir(outlinePicturePath)
        np.save(os.path.splitext(image)[0],canny2)
#轮廓精简
def outlineCut(outlinePicturePath,outlineCutPicturePath):
    dirs = os.listdir(outlinePicturePath)
    for image in dirs:
        print(image)
        os.chdir(outlinePicturePath)
        img = np.load(image)
        img2 = img
        ret, img2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(img2, cv2.CHAIN_APPROX_TC89_L1, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        area1 = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > area1:
                cv_contours.clear()
                area1 = area
                cv_contours.append(contour)
                # x, y, w, h = cv2.boundingRect(contour)
                # img[y:y + h, x:x + w] = 255
            else:
                continue
        img2.fill(255)
        img2 = cv2.drawContours(img2, cv_contours, -1, (0, 255, 0),1)
        cv2.namedWindow(image, cv2.WINDOW_NORMAL)
        cv2.imshow(image, img2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        os.chdir(outlineCutPicturePath)
        np.save(os.path.splitext(image)[0], img2)

def lineCut(outlineCutPicturePath,lineCutPicturePath):
    dirs = os.listdir(outlineCutPicturePath)
    for image in dirs:
        print("开始处理" + image)
        os.chdir(outlineCutPicturePath)
        edges = np.load(image)
        image1 = edges
        xy = getCorner(edges)
        print(xy)
        images = getLine(image1,xy)
        saveLine(image,lineCutPicturePath,images)
        print(image + "line保存完毕")
#保存图片
def saveLine(image,lineCutPicturePath,images):
    file_name = os.path.splitext(image)[0]
    folder = os.path.exists(lineCutPicturePath + "\\" + file_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(lineCutPicturePath + "\\" + file_name)  # makedirs 创建文件时如果路径不存在会创建这个路径

    else:
        print("目标路径已存在")
        "---  There is this folder!  ---"
    os.chdir(lineCutPicturePath + "\\" + file_name)
    i = 0
    for img in images:
        np.save("line" + str(i), img)
        i = i + 1


#得到线集
def getLine(gray,xy):
    image = gray
    images = []
    xy = remakePoint(gray,xy)
    print(xy)
    stx, sty = xy[0]
    while len(np.unique(gray)) != 1:
        gray,image,stx,sty,cx,cy = getOneLine(gray,xy,stx,sty)
        image = np.array(image)
        images.append(image)
        if [stx,sty] == [cx,cy]:
            break

    return images
#重定位角点
def remakePoint(gray,xy):
    #print(xy)
    Neighbors = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                 (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
                 (0, -2), (0, -1), (0, 1), (0, 2),
                 (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
                 (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]
    nxy = []
    for nnxy in xy:
        stx,sty = nnxy[0]
        for dr,dc in Neighbors:
            if gray[sty + dc][stx + dr] == 0:
                nxy.append([sty +dc,stx + dr])
                break

    #print(nxy)
    return nxy
#得到一条线
def getOneLine(gray,xy,x1,y1):
    Neighbors = [(-1,-1),(-1,0),(-1,1),
                 (0,-1),(0,1),
                 (1,-1),(1,0),(1,1)]
    stx = x1
    sty = y1
    image = np.zeros(shape=gray.shape)
    image.fill(255)
    nex = stx
    ney = sty

    image[stx][sty] = 0
    while [nex,ney] not in xy or [nex,ney] == [stx,sty]:
        lx = nex
        ly = ney
        for neighbor in Neighbors:
            dr, dc = neighbor
            if gray[nex + dr][ney + dc] == 0:
                image[nex + dr][ney + dc] = 0
                gray[nex + dr][ney + dc] = 255
                nex = nex + dr
                ney = ney + dc
                break
        if [nex,ney]  in xy:
            break
        if [nex,ney] == [lx,ly]:
            break

    return gray, image, nex, ney,stx,sty


#角点检测
def getCorner(gray):

    corners = cv2.goodFeaturesToTrack(gray, 20, 0.4, 20)
    corners = np.int0(corners)

    return corners


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    originalPath = "F:\\picture\\originalPicture"
    clipPath = "F:\\picture\\clipPicture"
    vectorPath = "F:\\picture\\vector"
    outlinePath = "F:\\picture\\outline"
    outlineCutPath = "F:\\picture\\outlineCut"
    lineCutPath = "F:\\picture\\lineCut"
    #reNamePicture(originalPath)
    getOutline(originalPath,outlinePath)
    outlineCut(outlinePath,outlineCutPath)
    lineCut(outlineCutPath,lineCutPath)
    #clipPicture(originalPath,clipPath)
    #pictureToVector(clipPath,vectorPath)
