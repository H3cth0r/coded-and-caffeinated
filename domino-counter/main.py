import cv2
import numpy as np
import sys

def count_dots_debug(image_path):
    # --- CONFIGURATION ---
    TARGET_WIDTH = 800
    MIN_AREA = 20           # Minimum dot size
    MAX_AREA = 2000         # Maximum dot size (dots are usually smaller than the full domino)
    MIN_SOLIDITY = 0.80     # How "solid" the shape is (excludes jagged wood grain)
    MIN_CONTRAST = 40       # Minimum darkness difference
    
    # 1. Load and Resize
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read '{image_path}'")
        return

    h, w = img.shape[:2]
    scale = TARGET_WIDTH / w
    img = cv2.resize(img, (TARGET_WIDTH, int(h * scale)))

    # --- DEBUG STEP 1: Preprocessing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Median blur is great for wood grain suppression
    blurred = cv2.medianBlur(gray, 5) 
    cv2.imwrite("debug_1_blurred.jpg", blurred)

    # --- DEBUG STEP 2: Thresholding ---
    # We use a larger block size (45) to handle the size of the domino tiles better
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 45, 10)
    cv2.imwrite("debug_2_raw_thresh.jpg", thresh)

    # --- DEBUG STEP 3: Morphological Separation ---
    # Erode allows separating touching dots.
    kernel = np.ones((3,3), np.uint8)
    thresh_separated = cv2.erode(thresh, kernel, iterations=1)
    cv2.imwrite("debug_3_separated.jpg", thresh_separated)

    # --- DEBUG STEP 4: Analysis & Filtering ---
    # CRITICAL CHANGE: cv2.RETR_LIST instead of RETR_EXTERNAL
    # RETR_EXTERNAL only finds the outer boundary of the domino.
    # RETR_LIST finds the outer boundary AND the dots inside.
    contours, _ = cv2.findContours(thresh_separated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    valid_dots = []
    
    # Visualization setup
    debug_analysis = img.copy()

    print(f"{'ID':<4} | {'Area':<6} | {'Solid':<5} | {'Circ':<5} | {'Ellip':<5} | {'Contr':<5} | {'Color':<5} | {'Status'}")
    print("-" * 85)

    for i, cnt in enumerate(contours):
        status = "Unknown"
        color_draw = (255, 255, 255) # Default White
        
        # 1. Area Check
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            # We don't draw extremely small/large noise to keep the image clean
            continue 

        # 2. Solidity Check (Filters out jagged wood grain)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 3. Shape Check (Hybrid Circle/Ellipse)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
        
        is_shape_valid = False
        ellipse_ratio = 0.0
        
        # A. Ellipse Fit (Handles angled dots)
        if len(cnt) >= 5:
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                ellipse_area = (np.pi * MA * ma) / 4
                if ellipse_area > 0:
                    ellipse_ratio = area / ellipse_area
                    # Allow 25% deviation from a perfect mathematical ellipse
                    if 0.75 < ellipse_ratio < 1.25:
                        is_shape_valid = True
            except:
                pass
        
        # B. Circle Fit (Handles small straight dots)
        if not is_shape_valid and circularity > 0.7:
            is_shape_valid = True

        # 4. Contrast & Polarity Check
        # Create a mask for the dot and a ring around it
        mask_dot = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask_dot, [cnt], -1, 255, -1)
        
        mask_dilated = cv2.dilate(mask_dot, kernel, iterations=4)
        mask_ring = cv2.subtract(mask_dilated, mask_dot)
        
        mean_dot = cv2.mean(gray, mask=mask_dot)[0]
        mean_bg = cv2.mean(gray, mask=mask_ring)[0]
        
        contrast = abs(mean_dot - mean_bg)
        
        # Polarity Check: Dots must be DARKER than their background (black paint vs white plastic)
        # mean_dot (Inside) should be lower than mean_bg (Outside)
        is_dark_dot = mean_dot < mean_bg

        # --- DECISION LOGIC ---
        if solidity < MIN_SOLIDITY:
            status = "Low Solidity"
            color_draw = (0, 0, 255) # Red
        elif not is_shape_valid:
            status = "Bad Shape"
            color_draw = (0, 0, 255) # Red
        elif not is_dark_dot:
            status = "Light Spot" # Reject white glare spots
            color_draw = (0, 255, 255) # Yellow
        elif contrast < MIN_CONTRAST:
            status = "Low Contrast" # Reject faint wood knots
            color_draw = (0, 255, 255) # Yellow
        else:
            status = "VALID"
            color_draw = (0, 255, 0) # Green
            valid_dots.append(cnt)

        # Draw contour
        cv2.drawContours(debug_analysis, [cnt], -1, color_draw, 2)
        
        # Draw ID
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(debug_analysis, str(i), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Console Log
        pol_str = "Dark" if is_dark_dot else "Light"
        print(f"{i:<4} | {int(area):<6} | {solidity:.2f}  | {circularity:.2f}  | {ellipse_ratio:.2f}  | {int(contrast):<5} | {pol_str:<5} | {status}")

    # Final Count
    print("-" * 85)
    print(f"Total Valid Dots: {len(valid_dots)}")
    
    cv2.imwrite("debug_4_analysis.jpg", debug_analysis)
    print("\n--- RESULTS ---")
    print("Saved 'debug_4_analysis.jpg'. Check the colors:")
    print("GREEN  = Detected Dot")
    print("RED    = Rejected (Shape is weird/jagged)")
    print("YELLOW = Rejected (Not a dark spot/Contrast too low)")

if __name__ == "__main__":
    image_file = "domino-4.jpg" 
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    count_dots_debug(image_file)
