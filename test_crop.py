import sys
sys.path.append("C:\\Users\\nagar\\.pyenv\\pyenv-win\\versions\\3.12.10\\Lib\\site-packages")
sys.path.append("C:\\Users\\nagar\\dolphin-training\\python-stubs")


from dolphin import event, savestate # type: ignore
from PIL import Image
import numpy as np
import cv2

await event.frameadvance()

savestate.load_from_slot(3)

await event.frameadvance()

def threshold(x):
    return 255 if x > 200 else 0

def is_white_rgb(img):

    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Mask bright white-ish regions
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white).astype(np.float32) / 255.0

    # Stronger boost for masked regions: scale [0.5, 2.0]
    enhanced = gray * (0.15 + 2.0 * mask)

    # Apply gamma correction to increase contrast
    gamma = 1.8  # higher = more aggressive contrast
    enhanced = 255 * ((enhanced / 255) ** gamma)

    # Clip and convert back
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return Image.fromarray(enhanced, mode='L')

j = 0
for i in range(200):
    if i % 4 == 0:
        (width, height, data) = await event.framedrawn()

        print(f"width: {width}, height: {height}")

        img = Image.frombytes("RGB", (width, height), data)
        
        crop_left = 316
        crop_size = 317 #456
        crop_right = crop_left + crop_size
        img1 = is_white_rgb(img)
        # img1.save(f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\img\\img{j}_full.png")
        img = img.crop((crop_left, height-crop_size, crop_right, height))
        img = img.resize((84, 84))
        # crop_size = 58
        # img = img.crop((4, 24, crop_size + 4, 84))
        # img = img.convert("L")
        # img.save(f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\img\\img{j}_rgb.png")
        # img1 = img1.crop((crop_left, 0, crop_right, height))
        # img1 = img1.resize((84, 84))
        # img1 = img1.crop((3, 25, crop_size + 3, 84))
        img1 = img1.crop((crop_left, height-crop_size, crop_right, height))
        img1 = img1.resize((84, 84))
        
        # img2 = img.convert("1", dither=Image.Dither.NONE)
        # img = img.point(threshold, mode="1")
        
        # img = isolate_ball_position_aware(img)
        img1.save(f"C:\\Users\\nagar\\dolphin-training\\python-stubs\\img\\img{j}_1.png")
        print(f"Image {j} saved")
        j += 1
    await event.frameadvance()


