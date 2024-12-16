import os
import logging
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import traceback

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('processed_grid', exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='sudoku_processor.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_sudoku_grid(image_path):
    """
    Detect Sudoku grid from uploaded image

    Args:
        image_path (str): Path to the uploaded image

    Returns:
        tuple: (detected grid image, success boolean)
    """
    try:
        # Read image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area
        grid_candidates = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) > 10000  # Minimum area threshold
        ]

        if not grid_candidates:
            logging.error("No suitable grid contours found")
            return None, False

        # Find the largest contour (most likely Sudoku grid)
        largest_contour = max(grid_candidates, key=cv2.contourArea)

        # Get bounding rectangle of grid
        x, y, w, h = cv2.boundingRect(largest_contour)
        grid_img = img[y:y + h, x:x + w]

        logging.info(f"Detected grid: position {x},{y} with size {w}x{h}")

        return grid_img, True

    except Exception as e:
        logging.error(f"Grid detection error: {str(e)}")
        logging.error(traceback.format_exc())
        return None, False


def split_grid_into_cells(grid_img):
    """
    Split the detected grid into 81 individual cells

    Args:
        grid_img (numpy.ndarray): Detected Sudoku grid image

    Returns:
        list: List of cell images
    """
    cell_height, cell_width = grid_img.shape[0] // 9, grid_img.shape[1] // 9
    cells = []

    for row in range(9):
        for col in range(9):
            # Compute cell coordinates
            y_start = row * cell_height
            x_start = col * cell_width

            cell = grid_img[
                   y_start:y_start + cell_height,
                   x_start:x_start + cell_width
                   ]

            # Crop 12% from each side
            crop_height = int(cell.shape[0] * 0.12)
            crop_width = int(cell.shape[1] * 0.12)

            cropped_cell = cell[
                           crop_height:-crop_height,
                           crop_width:-crop_width
                           ]

            # Resize to 28x28
            processed_cell = cv2.resize(cropped_cell, (28, 28))

            cells.append((f"{row}_{col}", processed_cell))

    return cells


def save_cells(cells, unique_id):
    """
    Save processed cells to disk

    Args:
        cells (list): List of processed cell tuples (name, image)
        unique_id (str): Unique identifier for this processing session
    """
    cell_dir = os.path.join('processed_grid', unique_id)
    os.makedirs(cell_dir, exist_ok=True)

    for cell_name, cell_img in cells:
        cv2.imwrite(os.path.join(cell_dir, f"{cell_name}.png"), cell_img)

    logging.info(f"Saved 81 cells for session {unique_id}")


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Main route to handle file uploads and processing
    """
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']

        # Check filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Detect and process grid
                grid_img, grid_detected = detect_sudoku_grid(filepath)

                if not grid_detected:
                    logging.error("Failed to detect Sudoku grid")
                    return render_template('index.html', error='Unable to detect Sudoku grid')

                # Split into cells
                cells = split_grid_into_cells(grid_img)

                # Save cells
                save_cells(cells, unique_id)

                return render_template('index.html', filename=filename)

            except Exception as e:
                logging.error(f"Processing error: {str(e)}")
                return render_template('index.html', error='Processing error')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)