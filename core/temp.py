import cv2
import matplotlib.pyplot as plt
import seaborn as sns


hm = cv2.imread('/home/francesco/Documents/Master-Thesis/core/maps/train/slope_rocks1.png')
hm = cv2.cvtColor(hm, cv2.COLOR_BGR2GRAY)

sns.heatmap(hm)
plt.showw