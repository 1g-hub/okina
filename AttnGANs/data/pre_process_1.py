import os

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        yield root
        for file in files:
            yield os.path.join(root, file)

file_list  = open('./filenames.txt','w+t')
img_list   = open('./images.txt','w+t')

img_id = 0

for file in find_all_files('./images'):
    if os.path.isfile(file):
        file_list.write(file[9:-4] + '\n')
        img_id += 1
        s = '%d %s\n'%(img_id, file[9:])
        img_list.write(s)

file_list.close()
img_list.close()
