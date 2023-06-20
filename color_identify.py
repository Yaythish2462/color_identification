from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
import os
import scipy.spatial.distance as dist
from PIL import Image
import argparse
import sys
from pathlib import Path
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def open_image(path):
    image = Image.open(path)
    return image

def display_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def RGBtoHEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image, number_of_colors, show_chart):          #function to get the colors from the image

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(image)
    
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGBtoHEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        explode = [0.05 for i in range(len(hex_colors))]
        fig1, ax1 = plt.subplots()
        ax1.pie(counts.values(), labels = hex_colors, colors = hex_colors, pctdistance = 0.85, explode = explode)
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        ax1.axis('equal')  
        plt.tight_layout()
        plt.show()
           
    return rgb_colors

def show_colors_in_image(image_array, rgb_color, threshold):      # function to show all the colors in the image
    white_image = np.zeros((image_array.shape), dtype=int)
    color = np.zeros((image_array.shape), dtype=int)
    for i in range(image_array.shape[0]):
        if dist.euclidean(image_array[i], rgb_color) <= threshold:
            white_image[i] = image_array[i]
        else:
            white_image[i] = [255, 255, 255]
        color[i] = rgb_color
    return white_image, color    

def make_plot(image_array, rgb_colors, height, width, plots):     # function to plot the colors in the image
    plt.figure(figsize=(20, 15))
    for i in range(1,plots*3+1):
        ax = plt.subplot(5, 6, i)
        if i%3 == 1:
            white_image, color = show_colors_in_image(image_array, rgb_colors[i//3], 40)
            plt.imshow(image_array.reshape(height, width, 3))
            plt.title('Original Image')
        if i%3 == 2:
            plt.imshow(white_image.reshape(height, width, 3))
            plt.title('Color in Image')
        if i%3 ==0:
            plt.imshow(color.reshape(height, width, 3))
            plt.title('Color')
        plt.axis("off")
        plt.tight_layout(pad=2.0)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--source', nargs='+', type=str, default=ROOT / 'Images/image0.jpg' , help='input the source of the model')
parser.add_argument('--dim', action='store_true', help='gives the dimension of the image')
parser.add_argument('--getcolors', action='store_true', help='To extract color from the image')
parser.add_argument('--getplot', action='store_true', help='To extract spectific color plot from the image')
args = parser.parse_args()    

path = (args.source)
image = open_image(path[0])
print("Opening image ......")
time.sleep(2)
display_image(image)
width = image.width
height = image.height

if args.dim:
    print("")
    print(".............................................")
    print("the width of the image is : " + str(width))
    print("the height of the image is : " + str(height))
    print(".............................................")


if args.getcolors:
    print("")
    print("Extracting color from the image")
    time.sleep(2)
    image_array = np.array(list(image.getdata()))
    image_array = image_array.reshape(width * height, 3)
    rgb_colors = get_colors(image_array, 10, True)
    print("done extracting")

if args.getplot:
    print("")
    print("getting the plot.......")
    make_plot(image_array, rgb_colors, height, width, 10)
    print("done plotting")

    




