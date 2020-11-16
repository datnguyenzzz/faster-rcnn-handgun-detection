import cv2

video_path = "D:\My_Code\database\images\\test\\truth-serum-johnny-english-funny-clip-mr-bean-official_0nBUGZHJ_SRyI.avi"
vcapture = cv2.VideoCapture(video_path)

h,w = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))

print(h,w)

fps = vcapture.get(cv2.CAP_PROP_FPS)

file_name = "detected_output.avi"
vwrite = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

count = 0
success = True

while success:
    print("frame: ", count)
    success, image = vcapture.read()

    if success:
        image = image[..., ::-1]
        print(image)

        count += 1
