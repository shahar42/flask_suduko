# Required Imports
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import pytesseract
from tensorflow.keras.models import load_model
import os

# Initialize Flask App
app = Flask(__name__)

# Load Pre-trained Digit Classifier Model
MODEL_PATH = r"C:\Users\peleg marelly\Desktop\Python\digit_recognizer.h5"  # Replace with your model path
model = load_model(MODEL_PATH)

# Define directory for saving invalid cells
INVALID_CELLS_DIR = "invalid_cells"
os.makedirs(INVALID_CELLS_DIR, exist_ok=True)

# Threshold for OCR Confidence (e.g., digits should be detected with at least 70% confidence)
OCR_CONFIDENCE_THRESHOLD = 0.71  # Adjust as needed

# Preprocess Image: Grayscale + Morphology + Binary Threshold
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel)
    # Increase the threshold for better separation of digits and blank spaces
    _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# Detect Grid: Using Morphological Line Detection
def detect_grid(binary):
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)))
    grid = cv2.add(horizontal, vertical)

    # Find grid bounding box
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_box = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_box)

# Extract Individual Cells: Without Perspective Warp
def extract_cells(image, bounding_box):
    x, y, w, h = bounding_box
    cell_size_x, cell_size_y = w // 9, h // 9
    cells = []

    for row in range(9):
        for col in range(9):
            x1, y1 = x + col * cell_size_x, y + row * cell_size_y
            cell = image[y1:y1 + cell_size_y, x1:x1 + cell_size_x]
            cells.append(((row, col), cell[5:-5, 5:-5]))  # Keep track of row, col
    return cells

# Recognize Digits: Hybrid OCR + CNN Classifier
def recognize_digits(cells, model):
    board = np.zeros((9, 9), dtype=int)
    metadata = []

    for (row, col), cell in cells:
        # Preprocess cell: Crop by 10% on each side
        height, width = cell.shape
        crop_x = int(width * 0.1)
        crop_y = int(height * 0.1)
        cropped_cell = cell[crop_y:height - crop_y, crop_x:width - crop_x]

        # Resize cropped cell to 28x28
        cell_resized = cv2.resize(cropped_cell, (28, 28))

        # Perform OCR
        ocr_text = pytesseract.image_to_string(cell_resized, config="--psm 10 digits")
        ocr_confidence = 1.0  # Default confidence is max for OCR

        if ocr_text.strip().isdigit():
            digit = int(ocr_text.strip())
        else:
            # Use CNN as fallback
            cell_norm = cell_resized / 255.0
            pred = model.predict(cell_norm.reshape(1, 28, 28, 1))
            digit = np.argmax(pred)

            # Calculate confidence of CNN prediction
            ocr_confidence = np.max(pred)

        # Only accept digit if it meets the confidence threshold (e.g., 70% confidence)
        if 1 <= digit <= 9 and ocr_confidence >= OCR_CONFIDENCE_THRESHOLD:
            # Valid digit, update board and metadata
            board[row][col] = digit
            metadata.append({"row": row, "col": col, "digit": digit})
        else:
            # Save invalid cells for debugging
            invalid_path = os.path.join(INVALID_CELLS_DIR, f"invalid_{row}_{col}.png")
            cv2.imwrite(invalid_path, cell)

    return board, metadata

# Solve Sudoku Board
def solve_sudoku(board):
    def is_valid(num, row, col):
        # Check row, column, and 3x3 grid
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        row_start, col_start = 3 * (row // 3), 3 * (col // 3)
        for i in range(row_start, row_start + 3):
            for j in range(col_start, col_start + 3):
                if board[i][j] == num:
                    return False
        return True

    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:  # Empty cell
                    for num in range(1, 10):
                        if is_valid(num, row, col):
                            board[row][col] = num
                            if backtrack():
                                return True
                            board[row][col] = 0  # Reset if not valid
                    return False  # If no valid number found, return False
        return True  # Puzzle is solved when no empty cells remain

    # Debug print to check the board before solving
    print("Initial Board (Before Solving):")
    print(board)

    if not backtrack():  # Run backtracking to solve the board
        print("No solution found!")
    else:
        print("Solved Board:")
        print(board)

    return board

# Prepare Full Sudoku Board
def prepare_board(image, model):
    binary = preprocess_image(image)
    grid_box = detect_grid(binary)
    cells = extract_cells(binary, grid_box)
    board, metadata = recognize_digits(cells, model)
    return board, metadata

# Convert NumPy arrays to native Python types for JSON serialization
def convert_to_native_types(obj):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalar types (like int64) to native Python types
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj  # Return other types unchanged

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

# Flask Routes
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'sudoku_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read uploaded file
    file = request.files['sudoku_image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Prepare the board
    board, metadata = prepare_board(image, model)
    solved_board = solve_sudoku(board)  # Get the solved board

    # Convert NumPy arrays to Python lists (which are JSON serializable)
    response = {
        "board": convert_to_native_types(board),  # Ensure conversion here
        "solved_board": convert_to_native_types(solved_board),  # Ensure conversion here
        "metadata": convert_to_native_types(metadata)  # Ensure conversion here
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)

