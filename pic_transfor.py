from PIL import Image
import os


filename = os.listdir('/home/szh/git/Image_Segmentation/dataset/test')
count =0
for i in filename:
    imgpath = os.path.join('/home/szh/git/Image_Segmentation/dataset/test',i)
    img = Image.open(imgpath)
    img1 = img.resize((512, 512))
    img2 = img1.convert('RGB')
    save_path = os.path.join('/home/szh/git/Image_Segmentation/dataset/test',(str(count)+'.png'))
    img2.save(save_path)
    count +=1


# img = Image.open('/home/szh/git/Image_Segmentation/dataset/test/11.png')
# img.show()
# print(len(img.split()))
# img1 = img.convert('RGB')
# img1.show()
# print(len(img1.split()))
# img1.save('/home/szh/git/Image_Segmentation/dataset/test/12.png')
