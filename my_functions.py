import math
import numpy as np
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

class myThreshold():
    '''
    二值化算法大全
    '''

    def getMinimumThreshold(self, imgSrc):
        """
        谷底最小值的阈值
        """
        Y = Iter = 0
        HistGramC = []
        HistGramCC = []

        #获取直方数组
        hist_cv = self.__getHisGram(imgSrc)

        for Y in range(256):
            HistGramC.append(hist_cv[Y])
            HistGramCC.append(hist_cv[Y])

        #通过三点求均值来平滑直方图
        while( self.__IsDimodal(HistGramCC) == False):
            HistGramCC[0] = (HistGramC[0] + HistGramC[0] + HistGramC[1]) / 3.0 #第一点
            for Y in range(1, 255):
                HistGramCC[Y] = (HistGramC[Y - 1] + HistGramC[Y] + HistGramC[Y + 1]) / 3 #中间的点

            HistGramCC[255] = (HistGramC[254] + HistGramC[255] + HistGramC[255]) / 3 #最后一点
            HistGramC = HistGramCC
            Iter += 1
            if (Iter >= 1000):
                return -1

        #阈值极为两峰之间的最小值
        Peakfound = False
        for Y in range(1, 255):
            if (HistGramCC[Y - 1] < HistGramCC[Y] and HistGramCC[Y + 1] < HistGramCC[Y]):
                Peakfound = True
            if (Peakfound == True and HistGramCC[Y - 1] >= HistGramCC[Y] and HistGramCC[Y + 1] >= HistGramCC[Y]):
                return Y - 1
        return -1

    def __IsDimodal(self, HistGram):
        #对直方图的峰进行计数，只有峰数位2才为双峰
        Count = 0

        for Y in range(1, 255):
            if HistGram[Y - 1] < HistGram[Y] and HistGram[Y + 1] < HistGram[Y]:
                Count += 1
                if(Count > 2):
                    return False

        if Count == 2:
            return True
        else:
            return False

    def __getHisGram(self, imgSrc):
        hist_cv = cv2.calcHist([imgSrc], [0], None, [256], [0, 256])
        return hist_cv

    def get1DMaxEntropyThreshold(self, imgSrc):
        """
        一维最大熵
        """
        X = Y = Amount = 0
        HistGramD = {}
        MinValue = 0
        MaxValue = 255
        Threshold = 0

        HistGram = self.__getHisGram(imgSrc)

        for i in range(256):
            if HistGram[MinValue] == 0:
                MinValue += 1
            else:
                break

        while MaxValue > MinValue and HistGram[MinValue] == 0:
            MaxValue -= 1

        if (MaxValue == MinValue):
            return MaxValue     #图像中只有一个颜色
        if (MinValue + 1 == MaxValue):
            return MinValue     #图像中只有二个颜色

        for Y in range(MinValue, MaxValue + 1):
            Amount += HistGram[Y]  #像素总数

        for Y in range(MinValue, MaxValue + 1):
            HistGramD[Y] = HistGram[Y] / Amount +1e-17

        MaxEntropy = 0.0
        for Y in range(MinValue + 1, MaxValue):
            SumIntegral = 0
            for X in range(MinValue, Y + 1):
                SumIntegral += HistGramD[X]

            EntropyBack = 0
            for X in range(MinValue, Y + 1):
                EntropyBack += (- HistGramD[X] / SumIntegral * math.log(HistGramD[X] / SumIntegral))

            EntropyFore = 0
            for X in range(Y + 1, MaxValue + 1):
                SumI = 1 - SumIntegral
                if SumI < 0:
                    SumI = abs(SumI)
                elif SumI == 0:
                    continue

                EntropyFore += (- HistGramD[X] / (1 - SumIntegral) * math.log(HistGramD[X] / SumI))

            if MaxEntropy < (EntropyBack + EntropyFore):
                Threshold = Y
                MaxEntropy = EntropyBack + EntropyFore

        if Threshold > 5:
            return Threshold - 5 #存在误差
        return Threshold


    def getIsoDataThreshold(self, imgSrc):
        """
        ISODATA （intermeans） 阈值算法
        :param imgSrc:
        :return:
        """

        HistGram = self.__getHisGram(imgSrc)
        g = 0
        for i in range(1, len(HistGram)):
            if HistGram[i] > 0:
                g = i + 1
                break

        while True:
            l = 0
            totl = 0
            for i in range(0, g):
                totl = totl + HistGram[i]
                l = l + (HistGram[i] * i)

            h = 0
            toth = 0
            for i in range(g+1, len(HistGram)):
                toth += HistGram[i]
                h += (HistGram[i] * i)

            if totl > 0 and toth > 0:
                l = l/totl
                h = h/toth
                if g == int((l + h / 2.0)):
                    break
            g += 1
            if g > len(HistGram) - 2:
                return 0

        return g

    def getIntermodesThreshold(self, imgSrc):

        HistGram = self.__getHisGram(imgSrc)
        return 126

    def getAlgos(self):
        """
        获取阈值算法
        :param index:
        :return:
        """

        algos = {
            0 : 'getMinimumThreshold',  #谷底最小值
            1 : 'get1DMaxEntropyThreshold', #一维最大熵
            2 : 'getIsoDataThreshold', #intermeans
            # 3 : 'getKittlerMinError', #kittler 最小错误
            4 : 'getIntermodesThreshold', #双峰平均值的阈值
        }

        return algos

def gray_to_binary(gray, method=1):
    '''灰度图像二值化
    '''
    j = method  # 选择阈值获取算法0,1,2,3,4,5
    thr = myThreshold()
    algos = thr.getAlgos()[j]  # 选择阈值获取算法0,1,2,3,4,5
    threshold = getattr(thr, algos)(gray)  # threshold = thr.get1DMaxEntropyThreshold(gray)

    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  # 输出：阈值、二值化数据
    return ret, binary

def cal_element_size(img):
    #根据图片大小粗略计算腐蚀 或膨胀所需核的大小
    sp = img.shape
    width = sp[1]  # width(colums) of image
    kenaly = math.ceil((width / 400.0) * 12)
    kenalx = math.ceil((kenaly / 5.0) * 4)
    a = (int(kenalx), int(kenaly))
    return a

def find_id_regions(img):
    '''查找身份证号码可能的区域列表'''
    regions = []
    # 1. 查找轮廓
    _,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if(area < 1000):
            continue

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # 计算高和宽 参考：http://blog.csdn.net/lanyuelvyun/article/details/76614872
        width = rect[1][0]
        hight = rect[1][1]

        # 筛选那些太细的矩形，留下扁的
        if hight > width:
            if hight < width * 5:
                continue
        else:
            if width < hight * 5:
                continue

        regions.append(rect)

    return regions

def is_identi_number(num):
    """
    检查是否为身份证号码
    :param num:
    :return:
    """
    if (len(num) != 18):
        return False
    Wi = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    Ti = ['1', '0', 'x', '9', '8', '7', '6', '5', '4', '3', '2']
    sum = 0
    for i in range(17):
        sum += int(num[i]) * Wi[i]
    if Ti[sum % 11] == num[-1].lower():
        return True
    else:
        return False

def crop_img_by_box(img, box):
    """
    通过顶点矩阵，裁剪图片
    :param imgSrc:
    :param box:
    :return:
    """

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    # 裁剪
    cropImg = img[y1:y1 + hight, x1:x1 + width]

    return cropImg, (x1, y1), width, hight

def find_chinese_regions(gray_img, id_rect):
    """
    根据身份证号码的位置推断姓名、性别、名族、出生年月、住址的位置
    :param cardNumPoint1: tuple 身份证号码所处的矩形的左上角坐标
    :param width: int 身份证号码所处的矩形的宽
    :param hight: int 身份证号码所处的矩形的高
    :return:
    """
    box = cv2.boxPoints(id_rect)  # 获取身份证号码位置信息
    box = np.int64(box)
    _, point, width, hight = crop_img_by_box(gray_img, box)  # 通过顶点获得身份证号码坐标信息

    # new_x = int(cardNumPoint1[0] - (width / 18) * 6)
    new_x = point[0] - (width / 18) * 5.5
    new_width = int(width / 5 * 4)

    box = []  # 通过身份证号码位置来推断其他区域位置
    # new_y = cardNumPoint1[1] - hight * 6.5
    card_hight = hight / (0.9044 - 0.7976)  # 身份证高度
    card_y_start = point[1] - card_hight * 0.7976  # 粗略算出图像中身份证上边界的y坐标

    # 为了保证不丢失文字区域，姓名的相对位置保留，以身份证上边界作为起始切割点
    # new_y = card_y_start# + card_hight * 0.0967

    # 容错因子，防止矩形存在倾斜导致区域重叠
    factor = 20

    new_y = card_y_start if card_y_start > factor else factor

    new_hight = card_hight * (0.7616 - 0.0967) + card_hight * 0.0967

    # 文字下边界坐标
    new_y_low = (new_y + new_hight) if (new_y + new_hight) <= point[1] - factor else point[1] - factor

    box.append([new_x, new_y])
    box.append([new_x + new_width, new_y])
    box.append([new_x + new_width, new_y_low])
    box.append([new_x, new_y_low])

    box = np.int64(box)
    return crop_img_by_box(gray_img, box)  # 获取汉字区域坐标信息，并剪切该区域

def horizontal_projection(binary_img):
    """
    水平投影边界坐标
    :return:
    """
    #水平行边界坐标
    boundaryCoors = []
    (x, y) = binary_img.shape
    a = [0 for z in range(0, x)]
    for i in range(0, x):
        for j in range(0, y):
            if binary_img[i, j] == 0:
                a[i] = a[i] + 1
                #BinaryImg[i, j] = 255  # to be white

    #连续区域标识
    continuouStartFlag = False
    up = down = 0
    tempUp = 0  #行高不足总高1/20,临时保存，考虑与下一个行合并。主要解决汉字中上下结构的单子行像素点不连续的问题

    for i in range(0, x):
        # for j in range(0, a[i]):
        #     BinaryImg[i, j] = 0

        if a[i] > 1 :
            if not continuouStartFlag:
                continuouStartFlag = True
                up = i
        else:
            if continuouStartFlag:
                continuouStartFlag = False
                down = i - 1
                if down - up >= x // 20 and down -up <= x//10:
                    #行高小于总高1/20的抛弃
                    boundaryCoors.append([up, down])
                else:
                    if tempUp > 0:
                        if down - tempUp >= x // 20 and down - tempUp <= x//10:
                            # 行高小于总高1/20的抛弃
                            boundaryCoors.append([tempUp, down])
                            tempUp = 0
                    else:
                        tempUp = up

    #print boundaryCoors
    #showImg(BinaryImg, 'BinaryImg')
    if len(boundaryCoors) < 4:
        return False

    return boundaryCoors

def get_id_nums(regions, gray_img):
    # 二值化处理
    ret, binary = gray_to_binary(gray_img, method=1)

    # 获得身份证号码
    cardNum=''
    angle = 0
    for rect in regions:
        angle = rect[2]
        # 高、宽、角度标定
        a, b = rect[1]
        if a > b:
            width = a
            hight = b
            pts2 = np.float32([[0, hight], [0, 0], [width, 0], [width, hight]])
        else:
            width = b
            hight = a
            angle = 90 + angle
            pts2 = np.float32([[width, hight], [0, hight], [0, 0], [width, 0]])

        # 透视变换（将图像投影到一个新的视平面）
        box = cv2.boxPoints(rect)
        pts1 = np.float32(box)  # 透视变换前位置
        M = cv2.getPerspectiveTransform(pts1, pts2)  #变换矩阵（根据透视前位置和透视后位置计算）
        cropImg = cv2.warpPerspective(binary, M, (int(width), int(hight)))  # 输出身份证号码区域透视后的图

        # 计算腐蚀和膨胀的核大小
        kenalx = kenaly = int(math.ceil((hight / 100.0)))
        # 膨胀和腐蚀操作
        kenal = cv2.getStructuringElement(cv2.MORPH_RECT, (kenalx, kenaly))
        dilation = cv2.dilate(cropImg, kenal, iterations=1)
        erosion = cv2.erode(dilation, kenal, iterations=1)

        #OCR识别
        cardNum = pytesseract.image_to_string(erosion)

        if not cardNum:
            continue

        if is_identi_number(cardNum):
            print('身份证有效')
            return cardNum, angle, rect
        else:
            print('无效身份证：%s'% cardNum)
            continue
    raise('身份证识别失败！！！')

def get_chinese_char(gray_char_area_img):
    """
    分析汉字区域，并识别提取
    :return:
    """
    # 二值化处理
    ret, binary = gray_to_binary(gray_char_area_img, method=1)

    # 2. 膨胀和腐蚀操作，得到可以查找矩形的图片
    # kenalx = kenaly = int(math.ceil((binary.shape[1] / 100.0))) # 计算腐蚀和膨胀的核大小
    # a= (kenalx , kenaly)
    a = cal_element_size(binary)  # 获取核大小
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, a)

    #微处理去掉小的噪点
    dilation_1 = cv2.dilate(binary, element1, iterations=1)
    erosion_1 = cv2.erode(dilation_1, element1, iterations=1)
    #文字膨胀与腐蚀使其连成一个整体
    erosion_2 = cv2.erode(erosion_1, element2, iterations=1)
    dilation_2 = cv2.dilate(erosion_2, element1, iterations=1)

    #获取各个文字行起始坐标
    boundaryCoors = horizontal_projection(dilation_2)
    if not boundaryCoors:
        raise('获取各个文字行起始坐标失败！')

    #垂直投影对行内字符进行切割
    textLine = 0 #有效文本行序号
    CARD_NAME = CARD_SEX = CARD_ETHNIC = CARD_YEAR = CARD_MON = CARD_DAY = CARD_ADDR = ''
    for textLine, boundaryCoor in enumerate(boundaryCoors):

        if textLine == 0:
            vertiCoors, text = get_name(binary, boundaryCoor,gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            CARD_NAME = text
        elif textLine == 1:
            vertiCoors, text = get_sex_ethic(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            CARD_SEX = text[0]
            CARD_ETHNIC = text[1]
        elif textLine == 2:
            #为了获取更加精准的值，通过身份证号码规则直接取得出生年月
            vertiCoors, text = get_birth(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            CARD_YEAR = text[0]
            CARD_MON = text[1]
            CARD_DAY = text[2]
        else:
            vertiCoors, text = get_address(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            CARD_ADDR += text

        # if DEBUG:
        #     fator = 2
        #     for verticoo in vertiCoors:
        #         box = [[verticoo[0] * scale - fator, boundaryCoor[0] * scale - fator],
        #                [verticoo[1] * scale + fator, boundaryCoor[0] * scale - fator],
        #                [verticoo[1] * scale + fator, boundaryCoor[1] * scale + fator],
        #                [verticoo[0] * scale - fator, boundaryCoor[1] * scale + fator],
        #                ]
        #         cv2.drawContours(img, [np.int64(box)], 0, (0, 255, 0), 2)

    return {'CARD_NAME':CARD_NAME, 'CARD_SEX':CARD_SEX, 'CARD_ETHNIC':CARD_ETHNIC,  'CARD_YEAR':CARD_YEAR, 'CARD_MON':CARD_MON ,'CARD_DAY': CARD_DAY, 'CARD_ADDR':CARD_ADDR}

def chars_cut(BinaryImg, horiBoundaryCoor):
    """
    文字通用切割处理
    :param BinaryImg:
    :param horiBoundaryCoor:
    :return:
    """
    # 列边界坐标
    vertiBoundaryCoors = []

    up, down = horiBoundaryCoor
    lineHight = down - up

    (x, y) = BinaryImg.shape
    a = [0 for z in range(0, y)]

    for j in range(0, y):
        for i in range(up, down):
            if BinaryImg[i, j] == 0:
                a[j] = a[j] + 1
                #BinaryImg[i, j] = 255  # to be white

    # 连续区域标识
    continuouStartFlag = False
    left = right = 0

    pixelNum = 0  # 统计每个列的像素数量
    maxWidth = 0  #最宽的字符长度
    for i in range(0, y):
        # for i in range((down - a[j]), down):
        #     BinaryImg[i, j] = 0
        pixelNum += a[i]  # 统计像素
        if a[i] > 0:
            if not continuouStartFlag:
                continuouStartFlag = True
                left = i
        else:
            if continuouStartFlag:
                continuouStartFlag = False
                right = i
                if right - left > 0:
                    if pixelNum > lineHight * (right - left) // 10:
                        curW = right - left
                        maxWidth = curW if curW > maxWidth else maxWidth
                        vertiBoundaryCoors.append([left, right])
                    pixelNum = 0  # 遇到边界，归零

    #showImg(BinaryImg, 'BinaryImgBinaryImg')
    return vertiBoundaryCoors, maxWidth

def _chineseCharHandle(BinaryImg, horiBoundaryCoor):
    # 获得该行字符边界坐标

    fator = 0.9

    vertiBoundaryCoors, maxWidth = chars_cut(BinaryImg, horiBoundaryCoor)
    newVertiBoundaryCoors = []  # 字符合并后的垂直系列坐标

    charNum = len(vertiBoundaryCoors)

    i = 0
    while i < charNum:
        if i + 1 >= charNum:
            newVertiBoundaryCoors.append(vertiBoundaryCoors[i])
            break

        curCharWidth = vertiBoundaryCoors[i][1] - vertiBoundaryCoors[i][0]
        if curCharWidth < maxWidth * fator:
            if vertiBoundaryCoors[i + 1][1] - vertiBoundaryCoors[i][0] <= maxWidth*(2 - fator):
                newVertiBoundaryCoors.append([vertiBoundaryCoors[i][0], vertiBoundaryCoors[i + 1][1]])
                i += 1
            elif curCharWidth > maxWidth / 4:
                newVertiBoundaryCoors.append(vertiBoundaryCoors[i])
        else:
            newVertiBoundaryCoors.append(vertiBoundaryCoors[i])

        i += 1
    return newVertiBoundaryCoors

def get_name(BinaryImg, horiBoundaryCoor, origin=None):
    """
    身份证姓名
    :param BinaryImg:
    :param horiBoundaryCoor:
    :param origin:
    :return:coors, text
    """

    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    if len((coors)) == 0:
        return coors, ''

    up, down = horiBoundaryCoor

    box = np.int64([[coors[0][0], up], [coors[-1][1], up], [coors[-1][1], down], [coors[0][0], down]])

    text = ''
    if type(origin) == np.ndarray:
        cropImg, _, _, _ = crop_img_by_box(origin, box)
        # OCR识别
        text = pytesseract.image_to_string(cropImg,'chi_sim', '7')

    return coors, text.replace(' ', '')

def get_sex_ethic(BinaryImg, horiBoundaryCoor, origin=None):
    """
    身份证性别 名族
    :param BinaryImg:
    :param horiBoundaryCoor:
    :param origin:
    :return:
    """
    text = ['', '']

    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    up, down = horiBoundaryCoor

    maxW = 0
    for coo in coors:
        curW = coo[1] - coo[0]
        maxW = curW if curW > maxW else maxW

    textIndex = 0
    if type(origin) == np.ndarray:
        for i in range(len(coors)):
            box = np.int64([[coors[i][0], up], [coors[i][1], up], [coors[i][1], down], [coors[i][0], down]])
            if (coors[i][1] - coors[i][0]) < maxW * 0.88:
                continue

            cropImg, _, _, _ = crop_img_by_box(origin, box)
            # OCR识别
            char = pytesseract.image_to_string(cropImg, 'chi_sim', '6')
            if textIndex == 0:
                text[0] = char  #性别
            else:
                if char == '民' or char == '族':
                    continue
                elif char == '又' or char == '汊':
                    text[1] += '汉'
                elif all(u'\u4e00' <= ch and ch <= u'\u9fff' for ch in char):
                    text[1] += char

            textIndex += 1

    #默认为汉族
    if len(text[1]) == 0:
        text[1] = '汉'

    return coors, text

def get_birth(BinaryImg, horiBoundaryCoor, origin=None):
    """
    身份证出生
    :param BinaryImg:
    :param horiBoundaryCoor:
    :return:
    """
    up, down = horiBoundaryCoor
    lineHight = down - up  # 字符高度

    vertiBoundaryCoors, maxWidth = chars_cut(BinaryImg, horiBoundaryCoor)
    newVertiBoundaryCoors = []  # 字符合并后的垂直系列坐标




    i = 0
    charNum = len(vertiBoundaryCoors)

    Section = [[] for j in range(charNum)]  # 按距离把字符分段
    sectIndex = 0

    while i < charNum:
        #当前字符宽度
        # curCharWidth = vertiBoundaryCoors[i][1] - vertiBoundaryCoors[i][0]
        if i+1 < charNum:
            rightDis = vertiBoundaryCoors[i+1][0] - vertiBoundaryCoors[i][1]
            if rightDis < 10:
                Section[sectIndex].append(vertiBoundaryCoors[i])
            else:
                Section[sectIndex].append(vertiBoundaryCoors[i])
                sectIndex += 1
        else:
            Section[sectIndex].append(vertiBoundaryCoors[i])

        i += 1

    retSection = []
    for i in range(len(Section)):
        if len(Section[i]) > 0:
            retSection.append(Section[i])

    # print vertiBoundaryCoors
    # print Section
    # print retSection

    YearCoor, index = _getYear(retSection, lineHight)
    yearLenght = YearCoor[1] - YearCoor[0]
    newVertiBoundaryCoors.append(YearCoor)

    month, index = _getMonth(retSection, index, yearLenght)
    if month:
        newVertiBoundaryCoors.append(month)

    DayCoor, index = _getDay(retSection, index, yearLenght)
    if DayCoor:
        newVertiBoundaryCoors.append(DayCoor)

    text = ['', '', '']

    # 减少ocr的调用，挺高性能， 只保留位置提取
    # if type(origin) == np.ndarray:
    #     for i in range(len(newVertiBoundaryCoors)):
    #         box = np.int0([[newVertiBoundaryCoors[i][0], up],
    #                        [newVertiBoundaryCoors[i][1], up],
    #                        [newVertiBoundaryCoors[i][1], down],
    #                        [newVertiBoundaryCoors[i][0], down]])
    #
    #         cropImg, _, _, _ = cropImgByBox(origin, box)
    #         chars = ocr(cropImg, 'eng', '7')
    #         text[i] = str(filterNonnumericChar(chars))
    return newVertiBoundaryCoors, text

def _getYear(Section, lineHight):
    _len = len(Section[0])
    if _len == 4:
        return [Section[0][0][0], Section[0][-1][1]], 0
    elif _len < 4:
        if Section[0][-1][1] - Section[0][0][0] > lineHight*3/2:
            return [Section[0][0][0], Section[0][-1][1]], 0
        else:
            _lenS = len(Section)
            for i in range(1, _lenS):
                if Section[i][-1][1] - Section[0][0][0] > lineHight * 2:
                    return [Section[0][0][0], Section[i][-1][1]], i
    else:
        return [Section[0][0][0], Section[0][-1][1]], 0

def _getMonth(Section, index, yearLenght):
    _lenS = len(Section)

    StartIndex = index + 1
    if StartIndex < _lenS:
        for i in range(StartIndex, _lenS):
            if Section[i][0][0] - Section[index][-1][1] < yearLenght *2/3:
                _leni = len(Section[i])
                if _leni > 1:
                    for j in range(1, _leni):
                        if Section[i][j][0] - Section[index][-1][1] > yearLenght *2/3:
                            _w = 0
                            for k in range(j, _leni):
                                _w += Section[i][k][1] - Section[i][k][0]
                                if _w < yearLenght//2:
                                    continue
                                else:
                                    if _w > yearLenght * 0.6:
                                        return [Section[i][j][0], Section[i][k-1][1]], i
                                    else:
                                        return [Section[i][j][0], Section[i][k][1]], i
                            return [Section[i][j][0], Section[i][-1][1]], i
                else:
                    continue
            else:
                if Section[i][-1][0] - Section[i][0][0] < yearLenght/4:
                    if i+1 < _lenS and Section[i+1][-1][1] - Section[i][0][0] <= yearLenght/2:
                        return [Section[i][0][0], Section[i+1][-1][1]], i+1
                    else:
                        return [Section[i][0][0], Section[i][-1][1]], i
                else:
                    return [Section[i][0][0], Section[i][-1][1]], i
    return False, StartIndex

def _getDay(Section, index, yearLenght):
    return _getMonth(Section, index, yearLenght)

def get_address(BinaryImg, horiBoundaryCoor, origin=None):
    """
    身份证地址
    :param BinaryImg:
    :param horiBoundaryCoor:
    :return:
    """

    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    up, down = horiBoundaryCoor

    box = np.int0([[coors[0][0], up], [coors[-1][1], up], [coors[-1][1], down], [coors[0][0], down]])

    text = ''
    if type(origin) == np.ndarray:
        cropImg, _, _, _ = crop_img_by_box(origin, box)
        # OCR识别
        text = pytesseract.image_to_string(cropImg, 'chi_sim', '7')

    return coors, text.replace(' ', '')


