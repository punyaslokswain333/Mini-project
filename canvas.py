# images 2- dimension
# grayscale (black and white or grey) we will deal
# colour (RGB)
import numpy as np
import cv2
from model import NeuralNet

net = Neuralnet()
net.show_performance()
 
canvas = np.ones(600,600), dtype="unit8" * 255
canvas[100:500,100:500] = 0
start_point = None
end_point = None
is_drawing = False
def draw_line(img,start_at,end_at,255,15):
    cv2.line(img,start_at,end_at,255,15)


# img = np.zeros([400, 400], dtype='uint8') * 255
# img[50:350, 50:350] = 0
# wname = 'canvas'
# cv2.namedWindow(wname)


def shape(event, x, y, flag, param):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point(x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP
        is_drawing = False

cv2.nameWindow(wname)
cv2.setMouseCallback(wname, shape)

while True:
    cv2.imshow(wname, canvas)
    key = cv2.waitKey(1)  # press any keyto go to next state

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[50:350, 50:350] = 0
    elif key == ord('w'):
        out = canvas[100:500, 100:500]
        result = net.predict(image)
        print("PREDICTION :", result)


cv2.destroyAllWindows()