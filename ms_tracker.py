import numpy as np
import cv2

from ex2_utils import Tracker, create_epanechnik_kernel, extract_histogram, backproject_histogram, get_patch, normalize_histogram

class MeanShiftTracker(Tracker):
    
    
    def initialize(self, image, region):
        self.bins = 16          #number of bins for the histogram
        self.alpha = 0.01       #learning rate for the histogram
        self.min_change = 0.1   #minimum change in the position
        self.max_iters = 50     #maximum number of iterations
        self.sigma = 0.5        #sigma for the epanechnikov kernel
    
        if len(region) == 8:
            x_ = np.array(region[::2]) #x_ and y_ are the coordinates of the bounding box
            y_ = np.array(region[1::2]) #x_ and y_ are the coordinates of the bounding box
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1] #region is the bounding box

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor #window is the size of the region of interest
        
        left = max(region[0], 0) #left is the x coordinate
        top = max(region[1], 0) #top is the y coordinate

        right = min(region[0] + region[2], image.shape[1] - 1) #right is the x coordinate
        bottom = min(region[1] + region[3], image.shape[0] - 1) #bottom is the y coordinate

        self.template = image[int(top):int(bottom), int(left):int(right)] #template is the region of interest
        # self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2) #center of the region
        
        self.position = [region[0] + region[2] / 2, region[1] + region[3] / 2 - 50] #center of the region
        #print(f"Position: {self.position}")
 
        #make kernel odd size since epanechnikov kernel is also odd
        ker_size_x = int(np.floor(region[2]))
        ker_size_y = int(np.floor(region[3]))
        if ker_size_x % 2 == 0:
            ker_size_x += 1
        if ker_size_y % 2 == 0:
            ker_size_y += 1
        
        self.size = (ker_size_x, ker_size_y)
        #print(f"Size: {self.size}")
        #create epanechnikov kernel
        self.e_kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.sigma) 

        #create the kerenl for mean shift
        small_x = (self.size[0] // 2) * -1
        small_y = (self.size[1] // 2) * -1

        self.x_kernel, self.y_kernel = np.meshgrid(np.arange(small_x, -small_x + 1), np.arange(small_y, -small_y + 1))
        
        # print(f"Kernel X: {np.shape(self.x_kernel)}")
        # print(f"Kernel Y: {np.shape(self.y_kernel)}")

    def mean_shift(self, image):
        #extract patch from the image
        patch, _ = get_patch(image, self.position, self.size) #patch is the region of interest in the image
        q_init = extract_histogram(patch, self.bins, self.e_kernel) #nov_q is the histogram of the patch
        iter = 0
        pos_history = [self.position]

        #iterate over the maximum number of iterations 
        while iter < self.max_iters:

            # extract histogram from the patch
            # from slides and article
            patch, _ = get_patch(image, self.position, self.size) #patch is the region of interest in the image
            p = extract_histogram(patch, self.bins) #p is the histogram of the patch
            v = np.sqrt(np.divide(q_init, (p + self.min_change ))) #v is the vector of the histogram
            w = backproject_histogram(patch, v , self.bins) #w is the backprojected histogram

            # calculate the mean shift move for x an y direction
            x_shift = np.sum((np.multiply(self.x_kernel, w) ) / np.sum(w))
            y_shift = np.sum((np.multiply(self.y_kernel, w) ) / np.sum(w))

            # break if the mean shift vector is smaller than the minimum change
            if abs(x_shift) < self.min_change and abs(y_shift) < self.min_change:
                break

            # Added this extra check since the above check was almost never enough to stop the algorithm
            if iter > 11:
                last_ten = pos_history[-10:]
                last_ten = np.array(last_ten)
                mean = np.mean(last_ten, axis=0)
                if abs(mean[0] - self.position[0]) < 0.5 and abs(mean[1] - self.position[1]) < 0.5:
                    #print("stopped because of mean")
                    break

            # posodobi histogram
            q_new = extract_histogram(patch, self.bins, self.e_kernel) #q is the histogram of the patch
            q_init = q_init * (1 - self.alpha) + q_new * self.alpha

            #update the position
            self.position = (self.position[0] + x_shift, self.position[1] + y_shift)
            
            pos_history.append(self.position)
            iter += 1

        # print(f"iterations: {iter}")


    def track(self, image):

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            # print("template is bigger than the image")
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        #mean shift part of the code
        self.mean_shift(image=image)

        
        return [self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]]
        # return [self.position[0], self.position[1], self.size[0], self.size[1]]

class MSParams():
	def __init__(self):
		self.enlarge_factor = 2