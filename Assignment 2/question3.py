import cv2
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FOLDER = Path('a2images')

main_image = DATA_FOLDER/'train.jpg' 
image_to_superimpose = DATA_FOLDER/'japanflag.webp' 
arch_image = cv2.imread(str(DATA_FOLDER/"Eiffelday.png"), cv2.IMREAD_COLOR)
flag_image = cv2.imread(str(DATA_FOLDER/"eiffelnight.png"), cv2.IMREAD_COLOR)

points = [] 
def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(arch_image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 Points", arch_image)

arch_image_rgb = cv2.cvtColor(arch_image, cv2.COLOR_BGR2RGB) 
fig, ax = plt.subplots()

plt.imshow(arch_image_rgb)
plt.title("Select 4 Points")
plt.axis('off')
points = plt.ginput(4) 
plt.show()

height, width, _ = flag_image.shape
src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
dst_points = np.array(points, dtype=np.float32)

homography_matrix, _ = cv2.findHomography(src_points, dst_points)

warped_flag = cv2.warpPerspective(flag_image, homography_matrix, (arch_image.shape[1], arch_image.shape[0]))

gray_warped_flag = cv2.cvtColor(warped_flag, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_warped_flag, 1, 255, cv2.THRESH_BINARY)

alpha = 0.5  # Transparency level (0.0 to 1.0)
warped_flag = cv2.addWeighted(warped_flag, alpha, arch_image, 1 - alpha, 0)

mask_inv = cv2.bitwise_not(mask)
arch_image_bg = cv2.bitwise_and(arch_image, arch_image, mask=mask_inv)
flag_fg = cv2.bitwise_and(warped_flag, warped_flag, mask=mask)

result_image = cv2.add(arch_image_bg, flag_fg)

# Step 12: Display the result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the map image and the flag image
# map_image = cv2.imread(str(DATA_FOLDER/"greatwall.jpg"), cv2.IMREAD_COLOR)
# flag_image = cv2.imread(str(DATA_FOLDER/"usflag.webp"), cv2.IMREAD_COLOR)

# # Resize the flag image to match the map image
# flag_resized = cv2.resize(flag_image, (map_image.shape[1], map_image.shape[0]))

# # Convert map image to grayscale
# map_gray = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)

# # Create a binary mask where the land (not white) is selected
# _, mask = cv2.threshold(map_gray, 250, 255, cv2.THRESH_BINARY_INV)  # Adjust threshold if necessary

# # Convert the mask to a 3-channel image to match the flag and map
# mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# # Create a 50% transparent version of the flag image
# alpha = 0.5
# flag_transparent = cv2.addWeighted(flag_resized, alpha, map_image, 1 - alpha, 0)

# # Apply the mask to the transparent flag (to keep it only on land)
# flag_on_land = cv2.bitwise_and(flag_transparent, mask_3ch)

# # Combine the original map image and the flag on land, ensuring the sea part remains unchanged
# final_image = cv2.add(cv2.bitwise_and(map_image, cv2.bitwise_not(mask_3ch)), flag_on_land)

# # Display the result
# plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
# plt.axis(False)
# plt.show()

