"""
Generate sample portrait photos with synthetic face patterns for quick demonstration.
"""
import os
import cv2
import numpy as np

def generate_sample_images(output_dir="samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample 1: Formal Portrait on neutral backdrop
    h, w = 900, 700
    img1 = np.full((h, w, 3), (235, 230, 225), dtype=np.uint8) # light gray/beige bg
    
    # Body / Shoulders
    cv2.ellipse(img1, (350, 850), (280, 260), 0, 0, 180, (50, 45, 60), -1) # dark blue suit
    # Shirt collar
    pts_shirt = np.array([[310, 620], [350, 680], [390, 620], [350, 750]])
    cv2.fillPoly(img1, [pts_shirt], (250, 250, 250))
    # Tie
    pts_tie = np.array([[342, 680], [358, 680], [365, 850], [335, 850]])
    cv2.fillPoly(img1, [pts_tie], (40, 40, 180)) # red tie
    # Neck
    cv2.rectangle(img1, (320, 500), (380, 620), (160, 190, 225), -1)
    # Head / Face
    cv2.ellipse(img1, (350, 430), (110, 145), 0, 0, 360, (170, 200, 235), -1) # skin tone
    # Hair
    cv2.ellipse(img1, (350, 330), (120, 70), 0, 180, 360, (30, 25, 25), -1)
    # Eyes
    cv2.circle(img1, (310, 410), 9, (40, 40, 40), -1)
    cv2.circle(img1, (390, 410), 9, (40, 40, 40), -1)
    cv2.circle(img1, (312, 408), 3, (255, 255, 255), -1)
    cv2.circle(img1, (392, 408), 3, (255, 255, 255), -1)
    # Nose
    cv2.line(img1, (350, 415), (347, 455), (140, 170, 205), 3)
    cv2.line(img1, (347, 455), (355, 455), (140, 170, 205), 3)
    # Smile
    cv2.ellipse(img1, (350, 485), (28, 14), 0, 0, 180, (100, 110, 190), 3)
    
    cv2.imwrite(os.path.join(output_dir, "portrait_formal.jpg"), img1, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    # Sample 2: Outdoor Portrait with colorful background
    h2, w2 = 1000, 800
    img2 = np.zeros((h2, w2, 3), dtype=np.uint8)
    for y in range(h2):
        ratio = y / h2
        img2[y, :] = (int(180 * (1 - ratio) + 80 * ratio), int(120 * (1 - ratio) + 160 * ratio), int(40 * (1 - ratio) + 60 * ratio))
    
    # Body / Shoulders
    cv2.ellipse(img2, (400, 950), (320, 280), 0, 0, 180, (180, 100, 40), -1) # teal jacket
    # Neck
    cv2.rectangle(img2, (365, 570), (435, 700), (150, 185, 220), -1)
    # Face
    cv2.ellipse(img2, (400, 490), (120, 155), 0, 0, 360, (160, 195, 230), -1)
    # Hair
    cv2.ellipse(img2, (400, 390), (135, 90), 0, 180, 360, (50, 80, 120), -1)
    # Eyes
    cv2.circle(img2, (355, 470), 10, (30, 30, 30), -1)
    cv2.circle(img2, (445, 470), 10, (30, 30, 30), -1)
    cv2.circle(img2, (357, 468), 3, (255, 255, 255), -1)
    cv2.circle(img2, (447, 468), 3, (255, 255, 255), -1)
    # Smile
    cv2.ellipse(img2, (400, 550), (35, 18), 0, 0, 180, (80, 90, 170), 4)
    
    cv2.imwrite(os.path.join(output_dir, "portrait_casual.jpg"), img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"Sample images created in '{output_dir}/'")

if __name__ == "__main__":
    generate_sample_images()
