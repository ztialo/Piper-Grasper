import cv2, numpy as np

# Example color ranges (H in [0..180] for OpenCV)
COLOR_RANGES = {
    "target":   [((0, 100, 100), (10,255,255)), ((170, 100, 100), (179, 255, 255))],  # red color
    "goal": [((35, 80,50), (85,255,255))], # green color
    "blue":  [((90, 80,50), (130,255,255))],
    "yellow":[((20,120,70), (32,255,255))],
    "mint":  [((160, 40, 150),  (175, 100, 255))]
}

# BGR drawing colors for visualization
DRAW = {"target":(139, 0, 0), "goal":(139, 0, 0), "blue":(255,0,0), "yellow":(0,255,255)}


def _find_square_contours(mask, min_rel_area=5e-4):
    H, W = mask.shape
    img_area = H*W

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_rel_area*img_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, max(2.0, 0.02*peri), True)
        if len(approx) < 4 or not cv2.isContourConvex(approx):
            continue

        # Fit minimal rectangle to be orientation-invariant
        rect = cv2.boxPoints(cv2.minAreaRect(approx))
        rect = rect.astype(np.float32)

        # Angles near 90°
        def ang(a,b,c):
            v1, v2 = a-b, c-b
            cosang = (v1@v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        angs = [ang(rect[(i-1)%4], rect[i], rect[(i+1)%4]) for i in range(4)]
        if not all(70 <= a <= 110 for a in angs):
            continue

        # Edge-length balance (square-ish)
        L = [np.linalg.norm(rect[i]-rect[(i+1)%4]) for i in range(4)]
        if min(L)/max(L) < 0.7:
            continue

        # Solidity (fills its hull)
        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        if area/hull_area < 0.9:
            continue

        kept.append(approx)

    return kept


def _distort_to_bird_eye(image):
    # pixel points collected from placing cube at desk four corners
    src = np.float32([[137, 125], [511, 125], [570, 440], [70, 440]])
    dst = np.float32([[0,0],[640,0],[640, 426],[0, 426]])

    # homography rotation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    width  = int(max(dst[:,0]))
    height = int(max(dst[:,1]))

    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def _cv_to_isaac_frame(wx, wy):
    wx_i = wy
    wy_i = -wx
    return wx_i, wy_i


def _pixel_to_isaac_coord(px, py):
    x_pxl_scaling_factor = 0.0014178
    y_pxl_scaling_factor = 0.0012105

    wx = px * x_pxl_scaling_factor
    wy = py * y_pxl_scaling_factor
    return round(wx, 3), round(wy, 3)


def _get_world_xy_estimation(bgr_warped, color_ranges=COLOR_RANGES):
    hsv = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)

    annotated = bgr_warped.copy()
    results = {c: [] for c in color_ranges}

    h, w, _ = bgr_warped.shape
    # world_origin = tuple(w/2, h)
    # print("bgr_warped world origin: ", world_origin)

    # To avoid double-counting overlapping colors, track a global 'taken' mask
    taken = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color, ranges in color_ranges.items():
        # combine all sub-ranges for this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # remove pixels already assigned to earlier colors (optional)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(taken))

        # find square-like faces
        polys = _find_square_contours(mask)

        # draw + collect
        for poly in polys:
            cv2.drawContours(annotated, [poly], -1, (255, 255, 0), 2)
            M = cv2.moments(poly)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                # convert to world frame where bottom center is the origin 
                wx = cx - (w/2)
                wy = h - cy
                px, py = _cv_to_isaac_frame(wx, wy)
                x_coord, y_coord = _pixel_to_isaac_coord(px, py)

                results[color].append({"x": x_coord, "y": y_coord, "contour": poly})

                cv2.circle(annotated, (cx, cy), 3, DRAW.get(color, (255,255,255)), -1)
                cv2.putText(annotated, color, (cx-30, cy-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW.get(color, (255,255,255)), 2, cv2.LINE_AA)
                text = f"({x_coord},{y_coord})"
                cv2.putText(annotated, text, (cx-90, cy+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW.get(color, (255,255,255)), 1, cv2.LINE_AA)

    return annotated, results


def _depth_around_centroid(depth_img, cx, cy, r=5):
    h, w = depth_img.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)

    patch = depth_img[y0:y1, x0:x1]

    # keep valid values (Isaac depth is meters; sometimes 0/inf can appear)
    vals = patch[np.isfinite(patch) & (patch > 0)]
    if vals.size == 0:
        return np.nan

    # robust estimate: median (top face interior is usually consistent)
    return float(np.median(vals))


def _get_world_z_estimation(raw_depth_value):
    z_scaling_factor = 0.01026
    z_w = raw_depth_value * z_scaling_factor
    return round(z_w, 3)

def detect_multi_color_cubes(rgba, depth, color_ranges=COLOR_RANGES):
    """Return annotated BGR image and a dict: color -> list of detections (cx, cy, contour)."""
    bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    bgr_warped = _distort_to_bird_eye(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    warped_annotated, warped_results = _get_world_xy_estimation(bgr_warped)
    # cv2.imshow("warped_annotated with world coord pixels", warped_annotated)

    annotated = bgr.copy()
    results = {c: [] for c in color_ranges}

    # To avoid double-counting overlapping colors, track a global 'taken' mask
    taken = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color, ranges in color_ranges.items():
        # combine all sub-ranges for this color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        # remove pixels already assigned to earlier colors (optional)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(taken))

        # find square-like faces
        polys = _find_square_contours(mask)
        # print(len(polys))



        # draw + collect
        for poly in polys:
            cv2.drawContours(annotated, [poly], -1, (255, 255, 0), 2)
            M = cv2.moments(poly)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.circle(annotated, (cx, cy), 3, DRAW.get(color, (255,255,255)), -1)
                cv2.putText(annotated, color, (cx-30, cy-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, DRAW.get(color, (255,255,255)), 2, cv2.LINE_AA)
                x_w = warped_results[color][0]["x"]
                y_w = warped_results[color][0]["y"]
                z_raw = _depth_around_centroid(depth, cx, cy, r=5)
                z_w = _get_world_z_estimation(z_raw)
                text = f"({x_w}, {y_w}, {z_w})"
                cv2.putText(annotated, text, (cx-60, cy+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW.get(color, (255,255,255)), 1, cv2.LINE_AA)
                results[color].append({"x_w": x_w, "y_w": y_w, "z_w": z_w, "contour": poly})

        # mark area as taken
        if polys:
            poly_mask = np.zeros_like(mask)
            cv2.drawContours(poly_mask, polys, -1, 255, thickness=cv2.FILLED)
            taken |= poly_mask

    return annotated, results
