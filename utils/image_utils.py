import cv2,os
def load_images(path,img_size):
    img_size=300
    img_list=[]
    labels=[]
    for i in os.listdir(path):
        dir_path = os.path.join(path, i)  # Use os.path.join for correct path construction
        for j in os.listdir(dir_path):
            img_path = os.path.join(dir_path, j)
            img = cv2.imread(img_path)
            if img is not None:
                resized_img = cv2.resize(img, (img_size, img_size))
                normalized_image=resized_img/255
                img_list.append(normalized_image)
                labels.append(i)
            else:
                print(f"Failed to load image: {img_path}") 
    return img_list, labels