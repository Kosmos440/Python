import cv2
import numpy as np
import requests
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
x_robo=1
y_robo=1
z_robo=1
X_SIZE=1000
Y_SIZE=1000
SIZE_ROBO = 64 #Половина основания (центр координат)
DIST_ROBO = 200 + SIZE_ROBO #Дистанция (перпендикуляр) от основания робота до листа
X_OFFSET = 0 #Относительно координат робота

def get_img():
    url = r'http://192.168.0.222/image.jpg'
    resp = requests.get(url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, -1)
    
def field_img():
    img = get_img()
    if 'res_img' not in globals():
        res_img = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)
    coords=[]
    if res[1] is not None and len(res[1])==4 and np.sum(res[1])==6:
        for i in range(4):
            marker=i
            index = np.where(res[1] == marker)[0][0]
            pt0 = res[0][index][0][marker].astype(np.int16)
            coords.append(list(pt0))
            cv2.circle(img, tuple(pt0), 10, (255,0,255), thickness=-1)
        height, width, _ = img.shape
        input_pt = np.array(coords)
        output_pt = np.array([[0, 0], [width, 0],[width, height],[0, height]])
        h, _ = cv2.findHomography(input_pt, output_pt)
        res_img = cv2.warpPerspective(img, h, (width, height))
    return (res_img)

def sphere_coord():
    img = get_img()
    pai_img = field_img()
    height, width, _ = img.shape
    hsv = cv2.cvtColor(pai_img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('h',hsv[:,:,0])
    #cv2.imshow('s',hsv[:,:,1])
    img_bin = cv2.inRange(hsv[:,:,0],0,70)
    kernel = np.ones((5, 5), 'uint8')
    img_bin = cv2.erode(img_bin, kernel, iterations=10)
    img_bin = cv2.dilate(img_bin, kernel, iterations=10)
    #cv2.imshow('bin',img_bin)
    contours, hierarchy = cv2.findContours( img_bin,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours( res_img, contours, -1, (255,0,0),
                        #3, cv2.LINE_AA, hierarchy, 1 )
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(pai_img, (x, y), (x+w, y+h), (255,0,0), 2)
        x_center = x + w // 2
        y_center = y + h // 2
        x_mm = 287*x_center // width
        y_mm = 200*y_center // height
        y_mm = 200-y_mm + DIST_ROBO
        if(x_mm <=287 + X_OFFSET//2):
            x_mm = x_mm-287//2 + X_OFFSET
        else:
            x_mm = x_mm-287//2+1 + X_OFFSET
        cv2.circle(pai_img, (x_center,y_center), 10, (128,120,255), thickness=-1)
        cv2.putText(pai_img, f"x: {x_mm}, y:{y_mm}",
                    (x_center,y_center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2, cv2.LINE_AA)
    return (x_mm, y_mm, pai_img)

def robot_field():
        #cv2.imshow('111',res_img)
    #cv2.imshow('image',cv2.hconcat([img, res_img]))
    res_img = sphere_coord()[2]
    res_img = cv2.resize(res_img,(287,200))
    m_img = np.zeros((X_SIZE, Y_SIZE,3), dtype='uint8')
    m_img[Y_SIZE//2-DIST_ROBO-200 : Y_SIZE//2-DIST_ROBO, X_SIZE//2-143+X_OFFSET : X_SIZE//2+144+X_OFFSET] = res_img
    cv2.arrowedLine(m_img, (X_SIZE//2,Y_SIZE//2), (X_SIZE//2,0), (0,0,255),5)
    cv2.arrowedLine(m_img, (X_SIZE//2,Y_SIZE//2), (X_SIZE,Y_SIZE//2), (0,0,255),5)
    cv2.putText(m_img, f"x",
                        (X_SIZE-20,Y_SIZE//2-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(m_img, f"y",
                        (X_SIZE//2+20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.circle(m_img, (500,500), 500, (255,210,0), 2)
    cv2.circle(m_img, (500,500), 64, (255,210,0), -1)
    return m_img
#    if cv2.waitKey(5) == 27:
#            break
#    cv2.destroyAllWindows()

def collage():
    a = get_img()
    b = sphere_coord()[2]
    c = robot_field()
    a = cv2.resize(a,(250,200))
    b = cv2.resize(b,(250,200))
    c = cv2.resize(c,(500,500))
    collage = cv2.hconcat([a, b])
    collage = cv2.vconcat([collage, c])
    cv2.imshow('collage', collage)
    cv2.waitKey(5)

while True:
    collage()
    print(sphere_coord()[0],sphere_coord()[1])




