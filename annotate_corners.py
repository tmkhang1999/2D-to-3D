import cv2
import numpy as np
import argparse
from pathlib import Path

# Store clicked points
clicked_points = []

def click_event(event, x, y, flags, param):
    """Mouse callback function to record clicked points."""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")
        clicked_points.append([x, y])
        # Draw a circle at the clicked point
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Annotate Corners', img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manually annotate chessboard corners in an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output .npy file.')
    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        exit()

    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # Display the image and set the mouse callback
    cv2.imshow('Annotate Corners', img)
    cv2.setMouseCallback('Annotate Corners', click_event)

    print("\nInstructions:")
    print("1. Click on the chessboard corners in a consistent order (e.g., left-to-right, top-to-bottom).")
    print("2. Make sure to click the inner corners only.")
    print(f"Expected number of corners: {7*7}") # Based on our previous check
    print("3. Press any key to finish annotation and save.")

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Convert points list to numpy array
    corners_2d = np.array(clicked_points, dtype=np.float32)

    if corners_2d.shape[0] != 7*7:
         print(f"Warning: Annotated {corners_2d.shape[0]} corners, but expected {7*7}.")
         print("Please re-run the script and annotate the correct number of corners.")
    else:
        # Save the annotated points
        np.save(str(output_path), corners_2d)
        print(f"\nAnnotated {corners_2d.shape[0]} corners. Saved to {output_path}")

    # Clean up windows
    cv2.destroyAllWindows() 