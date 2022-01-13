#
import cv2
import os.path

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
print(os.path.isfile(resource_path('haarcascade_frontalface_alt.xml')))


trainedface_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
on = True

while on == True:
    success, image = cam.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_coord = trainedface_data.detectMultiScale(grayimg)
    for (x, y, w, h) in face_coord:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("test", image)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        on = False

cam.release()

#scale = 50
#width = int(cam.shape[1]*scale/100)
#height = int(cam.shape[0]*scale/100)
#dim = (width,height)
#resize = cv2.resize(cam,dim,interpolation= cv2.INTER_AREA)





