import cv2
import easyocr

# Read the image file
image = cv2.imread('image4.jpg')

# Convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection
canny_edge = cv2.Canny(gray_image, 170, 200)

# Find contours based on edges
contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# Initialize license plate contour and x,y coordinates
contour_with_license_plate = None
license_plate = None
x, y, w, h = None, None, None, None

# Find the contour with 4 potential corners and create ROI around it
for contour in contours:
    # Find perimeter of contour and check if it's a closed contour
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    if len(approx) == 4:  # Check if it is a rectangle
        contour_with_license_plate = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = gray_image[y:y + h, x:x + w]
        break

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Perform license plate text recognition
results = reader.readtext(license_plate, detail=0)

# Extract the license plate text
text = ''.join(results)

# Draw license plate and write the text
image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
image = cv2.putText(image, text, (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)

print("License Plate:", text)

cv2.imshow("License Plate Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
