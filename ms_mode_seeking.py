import numpy as np
from ex2_utils import generate_responses_1, get_patch, generate_responses_custom
import matplotlib.pyplot as plt
import cv2
from matplotlib import image


def draw_path(image, pos_history):
    # image: input image
    # pos_history: list of positions (tuples) of the mean shift vector
    # Returns: image with path drawn on it

	for i in range(0, len(pos_history), 2):
		dot_pos = (round(pos_history[i][0]), round(pos_history[i][1]))
		img = cv2.circle(image, dot_pos, radius = 1, color = (0, 0, 0), thickness = -6)

	return img

def mean_shift(image, ker_size, min_change, start_pos, max_iters):
    # image: input image
    # ker_size: size of kernel (int) the kernel should be odd
    # min_change: minimum change in mean shift vector
    # start_pos: starting position of kernel (tuple)
    # max_iters: maximum number of iterations
    
    #kernel siz shuold be odd
    if ker_size % 2 == 0:
        print("kernel size should be odd")
        return None

    # Make square kernels for each direction (x, y)
    x_kernel = np.zeros((ker_size, ker_size))
    y_kernel = np.zeros((ker_size, ker_size))

    smallest_number = (ker_size // 2) * -1
    # print(smallest_number)

    # Fill kernels with values (from slides page 4)
    for i in range(ker_size):
            x_kernel[:, i] = smallest_number
            y_kernel[i, :] = smallest_number
            smallest_number += 1
    
    # print(x_kernel)
    # print(y_kernel)

    # Initialize variables for mean shift iteration
    pos_history = []
    pos_history.append(start_pos)
    pos = start_pos
    iters = 0

    # prejšnji_premik_x = 0
    # prejšnji_premik_y = 0
    # pred_prejšnji_premik_x = 0
    # pred_prejšnji_premik_y = 0

    # Iterate until minimum change or maximum number of iterations is reached
    while iters < max_iters:
        
        # Compute mean shift vector for current kernel
        patch , _ = get_patch(image, pos, (ker_size, ker_size))
        #slaba začetna pozicija in majhno jedro povzročita da so v patchu vse vrednosti 0 
        # kar povzroči deljenje z 0 pri izračunu premika

        premik_x = np.sum((x_kernel * patch)) / np.sum(patch)
        premik_y = np.sum((y_kernel * patch) / np.sum(patch))
        
        # Check if minimum change is reached
        # print(f"premik_x: {abs(premik_x)}, premik_y: {abs(premik_y)}")
        # if prejšnji_premik_x == premik_x and prejšnji_premik_y == premik_y and pred_prejšnji_premik_x == premik_x and pred_prejšnji_premik_y == premik_y:
        #     break

        if abs(premik_x) <= min_change or abs(premik_y) <= min_change:
            print("stopped because of min change")
            break

        # Added this extra check since the above check was almost never enough to stop the algorithm
        if iters > 11 and ker_size > 8:
            last_ten = pos_history[-10:]
            last_ten = np.array(last_ten)
            mean = np.mean(last_ten, axis=0)
            if abs(mean[0] - pos[0]) < 1 and abs(mean[1] - pos[1]) < 1:
                print("stopped because of mean")
                break

        # Update to new position
        pos = pos + np.array([premik_x, premik_y])

        #brute force stop condition for testing purposes
        # if position is within 1 of [50, 70]  stop 
        # if abs(pos[0] - 50) < 1 and abs(pos[1] - 70) < 1:
        #     break


        pos_history.append(pos)
        iters += 1

    return pos, iters, pos_history



# density_map = generate_responses_custom([70, 50], 1, [20, 70], 1.5, 15)
density_map = generate_responses_1()

start_position = [80,20]

end_pos, iterations, pos_history = mean_shift(density_map, 23, 0.01, start_position, 10000)
#pri teh parametrih pazi da manjše jedro potrebuje večjo natančnost da doseže končno pozicijo

print(f"Number of iterations: {iterations}")
print(f"End position: {end_pos}")
print(f"Found value: {density_map[round(end_pos[0])][round(end_pos[1])]}")
print(f"Actual value: {density_map[70][50]}")

a = np.array((end_pos[0], end_pos[1]))
b = np.array((50, 70))
print(f"euclidean distance: {np.linalg.norm(a - b)}")


narisani_koraki = draw_path(density_map, pos_history)
plt.imshow(narisani_koraki)
plt.show()


# TODO comparison of different kernal sizes, min change and starting positions
# compare 3x3, 5x5, 9x9, 21x21 and 41x41 kernel sizes
# compare 0.001, 0.01, 0.1, 1 min change
# compare (70, 50), (70, 20), (50, 20) starting positions
# Results should be presented in a table
