# code_#5 基于边缘的分割算法
 
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd


def edge_based_segmentation(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 膨胀操作，填充边缘
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)

    # 找到轮廓并绘制矩形框
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 50:  # 忽略面积较小的矩形框
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return img


def process_images(file_path, file_path2):
    data_dict = {}
    for file in os.listdir(file_path):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            img_path = os.path.join(file_path, file)
            img = cv2.imread(img_path)
            img_name = os.path.splitext(file)[0]
            new_img_name = img_name + '_method_2.jpg'
            new_img_path = os.path.join(file_path2, new_img_name)
            new_img = edge_based_segmentation(img)
            cv2.imwrite(new_img_path, new_img)

            # 计算平均RGB值和绿色部分像素数量
            image = Image.open(new_img_path)
            image_array = np.array(image)
            green_mask = np.all(image_array == [0, 255, 0], axis=-1) | np.all(image_array == [128, 255, 0], axis=-1)
            green_pixels = np.sum(green_mask)
            green_mean = np.mean(image_array[green_mask], axis=0)

            data_dict[img_name] = {'mean_rgb': green_mean.tolist(), 'green_pixels': green_pixels}

    # 将结果保存为Excel文件
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.to_excel(os.path.join(file_path2, 'results.xlsx'))


# 测试
file_path = ''  #文件输入路径
file_path2 = 'D:\\desktop\\Fredrik\\Supplemental_Data_Set_2\\method_#3_edge\\B_TAP'  #文件输出路径
process_images(file_path, file_path2)


# code_#4 基于阈值的分割算法
import cv2
import numpy as np
import os
import pandas as pd
import xlsxwriter

file_path = ""  # 输入文件夹路径
file_path2 = ""  # 输出文件夹路径
if not os.path.exists(file_path2):
    os.makedirs(file_path2)

# 定义绿色和黑色的阈值
green_thresh = 0
black_thresh = 0

# 定义结果字典
result_dict = {}

for filename in os.listdir(file_path):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # 读入图像
        img = cv2.imread(os.path.join(file_path, filename))

        # 预处理
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
        mask_green = cv2.inRange(hsv, (0, green_thresh, green_thresh), (80, 255, 255))
        mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, black_thresh))
        mask = cv2.bitwise_or(mask_green, mask_black)

        # 轮廓检测
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)

            # 框出轮廓
            x, y, w, h = cv2.boundingRect(max_contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 计算平均RGB值和绿色部分像素数量
            roi = img[y:y + h, x:x + w]
            r_avg, g_avg, b_avg = cv2.mean(roi)[:3]
            green_pixels = np.count_nonzero(mask_green[y:y + h, x:x + w])
            black_pixels = np.count_nonzero(mask_black[y:y + h, x:x + w])
            total_pixels = green_pixels + black_pixels

            # 将结果存入字典
            result_dict[filename] = [r_avg, g_avg, b_avg, total_pixels]

            # 保存结果图像
            cv2.imwrite(os.path.join(file_path2, filename[:-4] + "_method_2.jpg"), img)

# 将字典转化为DataFrame
df = pd.DataFrame.from_dict(result_dict, orient='index', columns=['R_avg', 'G_avg', 'B_avg', 'Green_pixels'])

# 输出到Excel
writer = pd.ExcelWriter('输出路径\\result.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
workbook = writer.book
worksheet = writer.sheets['Sheet1']
for idx, col in enumerate(df.columns):
    series = df[col]
    max_len = max((
        series.astype(str).map(len).max(),
        len(str(series.name))
    )) + 1
    worksheet.set_column(idx, idx, max_len)
writer.save()

print("处理完毕！")

# code_#3 基于形状和边缘的分割算法
import cv2
import os
import numpy as np
# 定义要读入图像的文件夹路径和要保存图像的文件夹路径
input_folder = "" #数据输入路径
output_folder = "" #数据输出路径
results = {}  #设置一个字典以存储所有的平均值和包含像素数
# 遍历文件夹中的所有图像文件
for file in os.listdir(input_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        # 读入图像
        img = cv2.imread(os.path.join(input_folder, file))


    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测算子检测边缘
    edges = cv2.Canny(blur, 50, 150)

    # 进行霍夫圆变换，检测圆形
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=12, param2=5, minRadius=3, maxRadius=6)
    #edges: 被应用霍夫圆变换算法的图像
    #cv2.HOUGH_GRADIENT: 使用霍夫梯度方法检测圆形
    #dp: 累加器图像的分辨率与原图之比的倒数
    #minDist: 检测到的圆之间的最小距离，如果距离小于该值，则被认为是同一圆，如果距离大于该值，则被认为是两个不同的圆
    #param1: Canny边缘检测器的高阈值，低阈值是高阈值的一半
    #param2: 用于确定圆心的阈值参数，它越小越容易检测出假圆
    #minRadius: 检测到的圆的最小半径
    #maxRadius: 检测到的圆的最大半径

    # 如果检测到了圆形，则在图像上画出红色圆形，并计算圆形内部像素的RGB平均值和所包含的像素点
    if circles is not None:
        # 将检测到的圆形坐标、半径转换为整数
        circles = np.round(circles[0, :]).astype("int")
        
        # 循环处理每个圆形
        for (x, y, r) in circles:
            # 在图像上画出圆形
            cv2.circle(img, (x, y), r+1, (0, 0, 255), 2)
            #注释：此处在霍夫圆变换算法检测到的圆形的半径基础上增加了1个像素，以作为补充，使得边缘没有被涵盖进去的绿色像素也纳入考量


            # 计算圆形内部像素的RGB平均值和所包含的像素点
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r+1, 255, -1)
            mean = cv2.mean(img, mask=mask)[:3]
            pixel_count = np.count_nonzero(mask)
            #print("RGB平均值：", mean)
            #print("像素点数：", pixel_count)  

            
            results[file] = {'mean': mean, 'pixel_count': pixel_count}

    # 输出标注红色圆形的图像
    #cv2.imshow("Output", img)
    # 保存图像到输出文件夹中
    output_file = os.path.splitext(file)[0] + "_circular.jpg"  # 将文件名后缀替换为_circular.jpg
    cv2.imwrite(os.path.join(output_folder, output_file), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


import pandas as pd
# 假设字典名为results
df = pd.DataFrame.from_dict(results, orient='index', columns=['mean', 'pixel_count'])

# 将DataFrame写入Excel文件
output_xlsx_file = "B_TAP_output.xlsx"
df.to_excel(os.path.join(output_folder+output_xlsx_file))

    
