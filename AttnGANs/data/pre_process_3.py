import cv2

with open("./filenames.txt", "r") as f:
    imgfilenames = ['./images/'+line.rstrip() + '.jpg' for line in f.readlines()]

bbox_list = open('./bounding_boxes.txt','w+t')
cnt = 0

for f in imgfilenames:
    im = cv2.imread(f)
    h = im.shape[0]
    w = im.shape[1]
    cnt += 1
    s = '%d 0 0 %d %d\n'%(cnt, w, h)
    bbox_list.write(s)
