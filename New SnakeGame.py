import math
import random
import cvzone
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)  # 0为自己的摄像头
cap.set(3, 1280)  # 宽
cap.set(4, 720)  # 高

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, pathFood):  # 构造方法
        self.points = []  # 蛇身上所有的点
        self.lengths = []  # 每个点之间的长度
        self.currentLength = 0  # 蛇的总长
        self.allowedLength = 150  # 蛇允许的总长度
        self.previousHead = 0, 0  # 第二个头结点

        # 加载食物图片，如果找不到则创建默认食物
        try:
            self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
            if self.imgFood is None:
                raise FileNotFoundError
        except:
            # 创建默认的红色圆形食物
            self.imgFood = np.zeros((50, 50, 4), dtype=np.uint8)
            cv2.circle(self.imgFood, (25, 25), 20, (0, 0, 255, 255), -1)
            print("使用默认食物图片")

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):  # 实例方法
        if self.gameOver:
            cvzone.putTextRect(imgMain, "Game Over", [300, 400],
                               scale=7, thickness=5, offset=20)
            cvzone.putTextRect(imgMain, f'Your Score:{self.score}', [300, 550],
                               scale=7, thickness=5, offset=20)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])  # 添加蛇的点列表节点
            distance = math.hypot(cx - px, cy - py)  # 两点之间的距离
            self.lengths.append(distance)  # 添加蛇的距离列表内容
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction 收缩长度
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the food 是否吃了食物
            rx, ry = self.foodPoint
            # 修复：使用正确的碰撞检测逻辑
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1
                print(f"得分: {self.score}")

            # Draw Snake 画蛇
            if self.points:
                # 绘制蛇身
                for i in range(1, len(self.points)):
                    point1 = tuple(map(int, self.points[i - 1]))
                    point2 = tuple(map(int, self.points[i]))
                    cv2.line(imgMain, point1, point2, (0, 0, 255), 20)

                # 绘制蛇头
                if len(self.points) > 0:
                    head_point = tuple(map(int, self.points[-1]))
                    cv2.circle(imgMain, head_point, 20, (200, 0, 200), cv2.FILLED)

            # Draw Food 画食物
            rx, ry = self.foodPoint
            imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                        (rx - self.wFood // 2, ry - self.hFood // 2))

            cvzone.putTextRect(imgMain, f'Score:{self.score}', [50, 80],
                               scale=3, thickness=3, offset=10)

            # Check for Collision 检查是否碰撞
            if len(self.points) > 5:  # 只有蛇足够长时才检查碰撞
                pts = np.array(self.points[:-10], np.int32)  # 排除靠近头部的点
                if len(pts) > 2:  # 确保有足够的点形成多边形
                    pts = pts.reshape((-1, 1, 2))
                    minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

                    if -1 <= minDist <= 1:
                        print("碰撞! 游戏结束")
                        self.gameOver = True

        return imgMain


# 创建游戏实例
game = SnakeGameClass("donut.png")

while True:
    success, img = cap.read()
    if not success:
        print("无法读取摄像头")
        break

    img = cv2.flip(img, 1)  # 将手水平翻转
    hands, img = detector.findHands(img, flipType=False)  # 左手是左手，右手是右手，映射正确

    if hands:
        lmList = hands[0]['lmList']  # hands是由N个字典组成的列表
        pointIndex = lmList[8][0:2]  # 只要食指指尖的x和y坐标
        img = game.update(img, pointIndex)

    cv2.imshow("Snake Game - Hand Control", img)
    key = cv2.waitKey(1)

    if key == ord('r'):  # 按R键重新开始游戏
        game.gameOver = False
        game.score = 0
        game.points = []
        game.lengths = []
        game.currentLength = 0
        game.allowedLength = 150
        game.randomFoodLocation()
        print("游戏重新开始")
    elif key == ord('q') or key == 27:  # 按Q或ESC退出
        break

cap.release()
cv2.destroyAllWindows()