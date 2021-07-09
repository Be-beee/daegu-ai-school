import cv2
import numpy as np
import json
import os

# image 읽기

# img = cv2.imread("dog.jpg")
# c_img = cv2.imread("cat.jpg")

# image save

# cv2.imwrite("copy_img.jpg", img)




# show image

# cv2.imshow('dog', img)
# cv2.waitKey() # 화면 띄운 후 기다림




# change to grayscale

# rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2는 BGR로 읽힌다. BGR -> RGB로 (색상 반전)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color -> 회색으로 convert
#
# cv2.imshow('', rgb_img)
# cv2.imshow('g', gray_img)
# cv2.waitKey()




# 색으로 보기

# (B, G, R) = cv2.split(img) # 각 색상을 채널로 말함
#
# color = R
# cv2.imshow("", color)
# cv2.waitKey()
#
#
# zeros = np.zeros(img.shape[:2], dtype="uint8") # 이미지 가로*세로 크기만큼 0을 채움
# print(zeros)
#
# cv2.imshow('Red', cv2.merge([zeros, zeros, R]))
# cv2.waitKey()
#
# cv2.imshow('Green', cv2.merge([zeros, G, zeros]))
# cv2.waitKey()
#
# cv2.imshow('Blue', cv2.merge([B, zeros, zeros]))
# cv2.waitKey()



# 픽셀 값 접근

# print(img[100, 200]) # 특정 위치의 픽셀값 받아오기
# cv2.imshow("", img)
# cv2.waitKey()
# cv2.destroyAllWindows()



# 크기 조절

# cv2.imshow("", img)
#
# img = cv2.resize(img, (400, 300)) # 크기 변경 -> 픽셀값이 깨질 수 있음
# cv2.imshow('big', img)
#
# img = cv2.resize(img, (100, 50))
# cv2.imshow('small', img)
#
# cv2.waitKey()



# 자르기

# cv2.imshow("", img[0:150, 0:100]) # y, x 순서
#
# cv2.imshow("change", img[100:150, 50:100])
# h, w, c = img.shape
#
# cv2.imshow("crop", img[int(h/2 - 50): int(h/2 + 50), int(w/2 - 50): int(w/2 + 50)])
# print(int(h/2 - 50), int(h/2 + 50), int(w/2 - 50), int(w/2 + 50)) # pixel 값은 int여야 한다 안 그럼 터짐
# cv2.waitKey()



# 도형 그리기

# line

# img = cv2.line(img, (100, 100), (180, 150), (0, 255, 0), 4) # 시작위치, 끝위치, 색상, 선굵기
# cv2.imshow("", img)
# cv2.waitKey()


# rectangle
# img = cv2.rectangle(img, (35, 26), (160, 170), (0, 255, 0), 3) # 시작위치(왼쪽 상단 점), 끝위치(오른쪽 하단 점), 색상, 선굵기
# cv2.imshow("", img)
# cv2.waitKey()




# circle

# img = cv2.circle(img, (200, 100), 30 , (0, 255, 0), 3) # 원의 중심점, 반지름, 색상, 선굵기
# img = cv2.circle(img, (200, 100), 30 , (0, 255, 0), -1) # 원의 중심점, 반지름, 색상, 선굵기(-1이면 내부 색상이 채워짐)
# cv2.imshow("", img)
# cv2.waitKey()
#


# poly
# pts = np.array([[35, 26], [35, 170], [160, 170], [190, 26]]) # numpy로 해야함
# img = cv2.polylines(img, [pts], True, (0, 255, 0), 3) # img, 도형(을 리스트형식으로 감싸서 다시 넣음), 모서리 마감처리 여부, 색상, 선굵기
# cv2.imshow("", img)
# cv2.waitKey()


# text

# img = cv2.putText(img, "dog", (200, 100), 0, 1, (255, 0, 0), 2) # 이미지, 글자, 시작위치, 폰트, 글자크기, 색상, 글자굵기
# cv2.imshow("", img)
# cv2.waitKey()




# 이미지 붙여넣기

# img = cv2.rectangle(img, (200, 100), (275, 183), (0, 255, 0), 2) # (x, y)
# cv2.imshow("", img)
# c_img = cv2.resize(c_img, (75, 83)) # 네모 안에 사진 붙여넣기 위해서 리사이징, (x, y)
# img[100: 183, 200: 275] = c_img # [y, x], 픽셀 위치에 고양이 이미지 대입
# cv2.imshow("change", img)
# cv2.waitKey()



# 이미지 더하기

# img = cv2.resize(img, (217, 232)) # 더할 이미지의 크기를 맞춰주기 위해
# add1 = img + c_img # 픽셀 값을 더하면서 값 변화가 너무 커져서 이미지가 다 깨짐
# add2 = cv2.addWeighted(img, float(0.5), c_img, float(0.5), 5) # 이미지 비중 조절해서 이미지를 더함
#
# cv2.imshow("1", add1)
#
# cv2.imshow("2", add2)
# cv2.waitKey()



# 이미지 회전

# height, width, c = img.shape
#
# img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # 시계방향 90도 회전
# img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # 반시계방향 90도 회전
# img180 = cv2.rotate(img, cv2.ROTATE_180)

# img_r = cv2.getRotationMatrix2D((width/2, height/2), 45, 1) # 회전의 중심점, 회전각도, 이미지 확대 비율


# cv2.imshow('90', img90)
# cv2.imshow('270', img270)
# cv2.waitKey()




# 이미지 반전

# img = cv2.flip(img, 0) # 이미지, 0=상하대칭, 1=좌우대칭
# cv2.imshow('270', img)
# cv2.waitKey()



# 이미지 아핀
# height, width, channel = img.shape
# matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 2) # 이미지가 아닌 매트릭스 값이 저장됨
# img = cv2.warpAffine(img, matrix, (width, height)) # 이미지, 매트릭스값, (너비, 높이)
#
# cv2.imshow('', img)
# cv2.waitKey()


# 이미지 밝기, 감마
# nimg = cv2.imread('night.jpg')
# table = np.array([((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]).astype("uint8") # 가중치를 제곱해서 감마값 구함
#
#
# gamma_img = cv2.LUT(nimg, table) # 감마값 계산해서 내줌
#
# val = 10 # randint(10, 50)
# array = np.full(nimg.shape, (val, val, val), dtype=np.uint8)
# all_array = np.full(nimg.shape, (30, 30, 30), dtype=np.uint8)
#
#
# bright_img = cv2.add(nimg, array).astype("uint8") # 원본 이미지에 각 픽셀당 50이 더해짐
# all_img = cv2.add(gamma_img, all_array).astype("uint8") # 감마가 추가된 이미지에 각 픽셀당 30이 더해짐
#
# cv2.imshow('original', nimg)
# cv2.imshow('all', all_img)
# cv2.imshow('bright', bright_img)
# cv2.imshow('gamma', gamma_img)
# cv2.waitKey()



# 이미지 블러링
# blur_img = cv2.blur(img, (15, 15)) # 전체 블러링 (이미지, (블러링 가로 가중치, 블러링 세로 가중치))
#
# roi = img[28:74, 95:165]
# roi = cv2.blur(roi, (15, 15)) # 일부 블러링
# img[28:74, 95:165] = roi
#
# cv2.imshow('blur', blur_img)
# cv2.imshow('s_blur', img)
# cv2.waitKey()




# 이미지 패딩

# img_pad = cv2.cv2.copyMakeBorder(img, 100, 100, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # top bottom left right 각 패딩을 더해준다.
# cv2.imshow("img_pad", img_pad)
# cv2.waitKey()

# img_padd = cv2.copyMakeBorder(img, 100, 100, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# cv2.imshow('pad2', img_padd)
# cv2.waitKey()



# cv2 cascade

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv2.imread('sample_face_1.jpeg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
# print(faces)
#
# cv2.imshow('face', img)
# cv2.waitKey()



# json write

# total_list = []
# for i in range(5):
#     ins_dic = {}
#     ins_dic[f"person{i}"] = i
#     ins_dic['bbox'] = [i+5, i+10, i+20, i+30]
#     total_list.append(ins_dic)
#
# with open('json_sample.json', 'w', encoding="utf-8") as make_file:
#     json.dump(total_list, make_file, indent="\t")




# json read

# json_dir = 'json_sample.json'
# print(os.path.isfile(json_dir))
# with open(json_dir) as f:
#     json_data = json.load(f)
#
# for i, j_data in enumerate(json_data):
#     print(j_data)
#     print(j_data['bbox'])






# practices 1

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img = cv2.imread('face.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
# print(faces)
#
# for (x, y, w, h) in faces:
#     img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
#
# cv2.imshow("face", img)
# cv2.waitKey()



# practices 2

# img = cv2.imread("test_img3.jpg")
# w, h, c = img.shape
#
# img_pad = cv2.copyMakeBorder(img, int((500-h)/2), int((500-h)/2), int((500-w)/2), int((500-w)/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])
#
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 1)
# print(faces)
#
# for (x, y, w, h) in faces:
#     img_pad = cv2.rectangle(img_pad, (x, y), (x+w, y+h), (0, 255, 0), 3)
#
# faces_list = []
# for i, (x, y, w, h) in enumerate(faces):
#     ins_dic = {}
#     ins_dic['object'] = f'person{i+1}'
#     ins_dic['box'] = [int(x), int(y), int(x+w), int(y+h)]
#     faces_list.append(ins_dic)
#
# with open('faces_data.json', 'w', encoding="utf-8") as make_file:
#     json.dump(faces_list, make_file, indent="\t")
#
#
# cv2.imshow("face", img_pad)
# cv2.waitKey()
#


# practice 3

img = cv2.imread("test_img3.jpg")
w, h, c = img.shape

img_pad = cv2.copyMakeBorder(img, int((500-h)/2), int((500-h)/2), int((500-w)/2), int((500-w)/2), cv2.BORDER_CONSTANT, value=[0, 0, 0])

json_dir = 'faces_data.json'
print(os.path.isfile(json_dir))
with open(json_dir) as f:
    json_data = json.load(f)

face_data_from_json = []
for i, j_data in enumerate(json_data):
    face_data_from_json.append(j_data['box'])
print(face_data_from_json)

for (x1, y1, x2, y2) in face_data_from_json:
    print(x1, y1, x2, y2)
    area = img_pad[y1: y2, x1: x2]
    area = cv2.blur(area, (15, 15))
    img_pad[y1: y2, x1: x2] = area

cv2.imshow('s_blur', img_pad)
cv2.waitKey()