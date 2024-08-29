import cv2

cap = cv2.VideoCapture(0)

while True:
    success, ret = cap.read()
    
    if not success:
        break
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Doing")
        cv2.imwrite('./image/image.jpg', ret)
    
    cv2.imshow('Frame', ret)   
    
cap.release()
cv2.destroyAllWindows()