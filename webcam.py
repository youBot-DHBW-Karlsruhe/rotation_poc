import cv2
import urllib.request
import numpy as np

while True:
    req = urllib.request.urlopen('http://192.168.0.27:8888/out.jpg')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'load it as it is'

    # gray = cv2.cvtColor(img, cv2.COLORMAP_RAINBOW)

    cv2.imshow('Stream', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done
cv2.destroyAllWindows()
