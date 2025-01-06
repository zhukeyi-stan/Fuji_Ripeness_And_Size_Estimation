import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label, size=None):
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    w, h = box[2] - box[0], box[3] - box[1]
    if label==0:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))
        if size is not None:
            size = np.around(size, decimals=2)
            ax.text(x0, y1, str(size), color='blue')
        else:
            ax.text(x0, y1, 'Ripe')
    elif label==1:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        if size is not None:
            size = np.around(size, decimals=2)
            ax.text(x0, y1, str(size), color='blue')
        else:
            ax.text(x0, y1, 'Unripe')


def save_fig(img,boxes,masks,labels,file_name="gsam_output.jpg",sizes=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    if sizes is not None:
        for box, label, size in zip(boxes, labels, sizes):
            try:
                show_box(box.numpy(), plt.gca(), label.numpy(), size)
            except:
                show_box(box, plt.gca(), label, size)
    else:
        for box, label in zip(boxes, labels):
            try:
                show_box(box.numpy(), plt.gca(), label.numpy())
            except:
                show_box(box, plt.gca(), label)

    plt.axis('off')
    if file_name.endswith('.eps'):
        plt.savefig(file_name,bbox_inches="tight", dpi=300, pad_inches=0.0, format='eps')
    else:
        plt.savefig(file_name,bbox_inches="tight", dpi=300, pad_inches=0.0)
    plt.close()

def save_clipped_fig(img,bboxes,img_name,save_dir):
    for i,bbox in enumerate(bboxes):
        x1,y1,x2,y2=bbox
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        clipped_img=img[y1:y2,x1:x2,:]
        cv2.imwrite(os.path.join(save_dir,img_name.replace('.png',f'({i}.png')),clipped_img)