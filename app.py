from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def warp_and_split_sudoku(image_path, output_folder):
    """
    Detect the Sudoku grid and divide it into 81 equal squares based on the outer thick lines.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to load image at '{image_path}'.")

    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        raise ValueError("Error: No contours found.")

    # Detect the largest quadrilateral (outer grid)
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            largest_contour = approx
            break
    else:
        raise ValueError("Error: Could not detect a quadrilateral Sudoku grid.")

    # Reorder points for proper perspective transformation
    def reorder_points(pts):
        pts = pts.reshape(4, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(sum_pts)],  # Top-left
            pts[np.argmin(diff_pts)],  # Top-right
            pts[np.argmax(sum_pts)],  # Bottom-right
            pts[np.argmax(diff_pts)],  # Bottom-left
        ], dtype="float32")

    ordered_pts = reorder_points(largest_contour)

    # Define the side length of the grid
    side = 450  # Pixels for the normalized grid
    dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype="float32")

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
    warped = cv2.warpPerspective(img, matrix, (side, side))

    grid_size = 9
    square_size = side // grid_size

    # Divide the warped image into 81 squares
    squares = []
    for row in range(grid_size):
        for col in range(grid_size):
            x_start = col * square_size
            y_start = row * square_size
            x_end = x_start + square_size
            y_end = y_start + square_size
            square = warped[y_start:y_end, x_start:x_end]
            squares.append(square)

    return squares

def generate_preview_image(squares, grid_size=9, square_size=50):
    """
    Combines 81 squares into a single preview image.
    """
    grid_image = np.zeros((grid_size * square_size, grid_size * square_size, 3), dtype=np.uint8)

    for row in range(grid_size):
        for col in range(grid_size):
            square = squares[row * grid_size + col]
            resized_square = cv2.resize(square, (square_size, square_size))
            if len(resized_square.shape) == 2:
                resized_square = cv2.cvtColor(resized_square, cv2.COLOR_GRAY2BGR)
            x_start = col * square_size
            y_start = row * square_size
            grid_image[y_start:y_start+square_size, x_start:x_start+square_size] = resized_square

    return grid_image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Process the file upload and return the processed image
        if 'file' not in request.files:
            return render_template("error.html", message="No file uploaded."), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = temp_dir.name

            # Save the uploaded file
            file_path = os.path.join(temp_path, filename)
            file.save(file_path)

            # Process the image and generate squares
            try:
                squares = warp_and_split_sudoku(file_path, temp_path)
            except ValueError as e:
                return render_template("error.html", message=str(e)), 400

            # Generate a preview image
            preview_image = generate_preview_image(squares)
            _, buffer = cv2.imencode(".jpg", preview_image)
            preview_bytes = BytesIO(buffer.tobytes())

            # Send the preview image as a response
            return send_file(preview_bytes, mimetype="image/jpeg")
        else:
            return render_template("error.html", message="Invalid file type. Only PNG, JPG, and JPEG are allowed."), 400
    else:
        # Handle GET request (render the form)
        return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    if 'file' not in request.files:
        return render_template("error.html", message="No file uploaded."), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = temp_dir.name

        # Save the uploaded file
        file_path = os.path.join(temp_path, filename)
        file.save(file_path)

        # Process the image and generate squares
        try:
            squares = warp_and_split_sudoku(file_path, temp_path)
        except ValueError as e:
            return render_template("error.html", message=str(e)), 400

        # Generate a preview image
        preview_image = generate_preview_image(squares)
        _, buffer = cv2.imencode(".jpg", preview_image)
        preview_bytes = BytesIO(buffer.tobytes())

        # Send the preview image as a response
        return send_file(preview_bytes, mimetype="image/jpeg")

    return render_template("error.html", message="Invalid file type. Only PNG, JPG, and JPEG are allowed."), 400

if __name__ == "__main__":
    app.run(debug=True)
