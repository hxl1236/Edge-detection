import cv2
import numpy as np

def merge_contours(bright_contours, dark_contours, max_neighbor_distance):
    merged_bright_contours = []
    unmerged_dark_contours = []
    unmerged_bright_contours = []

    for bright_contour in bright_contours:
        merged_bright = bright_contour
        merged = False
        updated_dark_contours = []  # New list to store dark contours without the contour being merged
        for dark_contour in dark_contours:
            merge_flag = False  # Flag to check if the dark contour is merged
            for point in dark_contour[:, 0]:
                pt = (int(point[0]), int(point[1]))  # Convert coordinates to integers
                dist = cv2.pointPolygonTest(bright_contour, pt, True)
                if dist > 0 and dist < max_neighbor_distance:
                    merged_bright = cv2.convexHull(np.concatenate((merged_bright, dark_contour)))
                    merge_flag = True
                    merged = True
                    break
            if not merge_flag:  # If the contour is not merged, add it to the updated dark contours list
                updated_dark_contours.append(dark_contour)
        dark_contours = updated_dark_contours  # Update dark_contours array after removing the merged contour
        if merged:
            merged_bright_contours.append(merged_bright)
        else:
            unmerged_bright_contours.append(bright_contour)

    unmerged_dark_contours = dark_contours
    return merged_bright_contours, unmerged_bright_contours, unmerged_dark_contours


def darkness(img, min_contour_area_threshold, max_contour_area_threshold):

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adjust contrast and brightness levels in the grayscale image
    alpha = 17
    beta = 10
    high_contrast_result = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
    img_blur = cv2.GaussianBlur(high_contrast_result, (7, 7), 0)
    cv2.imshow('dark', img_blur)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges_result = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image
    mask = np.zeros_like(edges_result)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Dilate the mask to connect nearby regions
    dilated_mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if min_contour_area_threshold <= cv2.contourArea(contour) <= max_contour_area_threshold]
    num_landmarks = len(filtered_contours)
    print("Number of Dark Landmarks:", num_landmarks)

    # Drawing the contours on the original image
    img_with_dark_contours = img.copy()
    cv2.drawContours(img_with_dark_contours, filtered_contours, -1, (0, 255, 0), 2)  # Draw green contours
    return img_with_dark_contours, filtered_contours

def brightness(img, min_contour_area_threshold, max_contour_area_threshold):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img_gray = 255 - img_gray

    # Adjust contrast and brightness levels in the inverted grayscale image
    alpha = 3
    beta = 40
    high_contrast_result = cv2.convertScaleAbs(inverted_img_gray, alpha=alpha, beta=beta)
    img_blur = cv2.GaussianBlur(high_contrast_result, (1, 1), 0)
    cv2.imshow('dark', img_blur)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges_result = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    contours, _ = cv2.findContours(edges_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(edges_result)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Dilate the mask to connect nearby regions
    dilated_mask = cv2.dilate(mask, None, iterations=3)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [contour for contour in contours if min_contour_area_threshold <= cv2.contourArea(contour) <= max_contour_area_threshold]
    num_landmarks = len(filtered_contours)
    print("Number of Bright Landmarks:", num_landmarks)

    # Drawing the contours on the original image
    img_with_bright_contours = img.copy()
    cv2.drawContours(img_with_bright_contours, filtered_contours, -1, (255, 0, 0), 2)  # Draw blue contours
    return img_with_bright_contours, filtered_contours

if __name__ == "__main__":
    img = cv2.imread('cutwhole1.tif')
    h, w, c = img.shape
    img = cv2.resize(img, (w*2,h*2))
    #min_dark_contour_area_threshold = int(input("Enter the minimum contour area threshold for dark regions: "))
    #max_dark_contour_area_threshold = int(input("Enter the maximum contour area threshold for dark regions: "))
    min_dark_contour_area_threshold = 50 #200
    max_dark_contour_area_threshold = 1500 #2500
    dark_result, dark_contours = darkness(img, min_dark_contour_area_threshold, max_dark_contour_area_threshold)
    cv2.imshow('dark', dark_result)
    cv2.waitKey(0)

    #min_bright_contour_area_threshold = int(input("Enter the minimum contour area threshold for bright regions: "))
    #max_bright_contour_area_threshold = int(input("Enter the maximum contour area threshold for bright regions: "))
    min_bright_contour_area_threshold = 200 #260
    max_bright_contour_area_threshold = 600 #1000
    bright_result, bright_contours = brightness(img, min_bright_contour_area_threshold, max_bright_contour_area_threshold)
    cv2.imshow('bright', bright_result)
    cv2.waitKey(0)

    # Merge contours
    max_neighbor_distance = 5
    merged_bright_contours, unmerged_bright_contours, unmerged_dark_contours = merge_contours(bright_contours, dark_contours, max_neighbor_distance)

    # Combine all contours into a single list
    all_contours = merged_bright_contours + unmerged_bright_contours + unmerged_dark_contours

    # Filter contours based on minimum area threshold
    filtered_contours = [contour for contour in all_contours if cv2.contourArea(contour) >= 50 and cv2.contourArea(contour) < 2000]#570
    overlaid_img = img.copy()
    num_landmarks = len(filtered_contours)
    print("Number of Landmarks:", num_landmarks)
    cv2.drawContours(overlaid_img, filtered_contours, -1, (0, 255, 0), 2)  # Draw filtered contours
    cv2.imshow('Overlayed Result', overlaid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
