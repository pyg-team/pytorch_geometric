from skimage.draw import line, circle_perimeter
from scipy import misc
import numpy as np
import math
import matplotlib.pyplot as plt

# Colormap
cmap = plt.cm.get_cmap('jet')
# Display partition borders
show_borders = True
# Output image size (x and y)
imagesize = 1200

# Random generated test filter
testfilter = np.random.uniform(-1.0, 1.0, size=[4, 8])
filter = testfilter


radius = int(imagesize/2-imagesize/12)
mid_x = mid_y = int(imagesize/2)
img = np.zeros((imagesize, imagesize), dtype=np.uint8)
img.fill(255)

img = np.expand_dims(img,axis=2)
img_col = np.concatenate([img,img,img],axis=2)

incircle_indices = [(x,y) for x in range(imagesize) for y in range(imagesize) if math.sqrt(((x-mid_x))**2+((y-mid_y))**2) < radius]


for (x,y) in incircle_indices:
    if (x,y) != (mid_x,mid_y):
        # Compute polar coordinates
        norm_x = x-mid_x
        norm_y = y-mid_y
        polar_r = (math.sqrt(norm_x**2 + norm_y**2)/radius) * (filter.shape[0] - 1)
        polar_theta = (math.atan2(norm_y,norm_x)+math.pi)/(math.pi*2) * filter.shape[1]

        # Compute control points and uv's
        u_p_1 = int(polar_r)
        u_p_2 = math.ceil(polar_r)
        u = polar_r-int(polar_r)
        v = polar_theta - int(polar_theta)
        v_p_1 = int(polar_theta) % filter.shape[1]
        v_p_2 = math.ceil(polar_theta) % filter.shape[1]

        # Sampling
        color = filter[u_p_1,v_p_1]*(1-u)*(1-v) \
                + filter[u_p_1,v_p_2]*(1-u)*v \
                + filter[u_p_2,v_p_1]*u*(1-v) \
                + filter[u_p_2,v_p_2]*u*v
        color = (color+1.0)/2.0
        rgba = np.array(cmap(color))
        img_col[x,y,:] = np.int8(rgba[:3]*255.0)


if show_borders:
    rr, cc = circle_perimeter(mid_x, mid_y, radius)
    img_col[rr, cc, :] = 0

    rr, cc = circle_perimeter(mid_x, mid_y, 1)
    img_col[rr, cc, :] = 0

    for i in range(1,filter.shape[0]-1):
        rr, cc = circle_perimeter(mid_x, mid_y, int(i*(radius/(filter.shape[0]-1))))
        img_col[rr, cc, :] = 0

    for j in range(filter.shape[1]):
        theta = j*(2.0*math.pi/filter.shape[1])
        rr, cc = line(mid_x, mid_y, int(mid_x + radius *math.cos(theta)), int(mid_y+radius*math.sin(theta)))
        img_col[rr, cc, :] = 0

    misc.imshow(img_col)
