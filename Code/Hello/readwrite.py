# 1 引入opencv 2 API 3 stop
# 1 文件的读取 2 封装格式解析 3 数据解码 4 数据加载
import cv2
img = cv2.imread('image/1.png', 1) # 0 gray 1 color

# cv2.imwrite('image/2.png', img) # 写图片

# 设置图片的质量
# JPG 压缩比范围0 - 100 ，有损压缩
cv2.imwrite('image/2.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])

# PNG 压缩比范围0-9 1 无损 2 透明度属性 
cv2.imwrite('image/3.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# cv2.imshow('WindowName', img) # 1 窗口名称 2 图片

# jpg png 1 文件头 2 文件数据

cv2.waitKey(0)