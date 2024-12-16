from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

SAVE_DIR = r"C:\Users\peleg marelly\Desktop\soduku solver\sudoku squares"
os.makedirs(SAVE_DIR, exist_ok=True)
# Initialize the Flask app
app = Flask(__name__)

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """
    Check if uploaded file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_square(square, size=(28, 28)):
    """
    Resize a single square to the given size (default is 28x28).
    """
    resized_square = cv2.resize(square, size, interpolation=cv2.INTER_AREA)
    return resized_square


# def create_combined_image(squares, grid_size=9):
#     """
#     Combine 81 resized 28x28 squares into a single 9x9 visualization image.
#     """
#     # Create a blank image of appropriate size
#     combined_image = np.zeros((grid_size * 28, grid_size * 28), dtype=np.uint8)
#
#     for row in range(grid_size):
#         for col in range(grid_size):
#             square = squares[row * grid_size + col]
#             # Place resized squares into their respective positions on the combined image
#             combined_image[row * 28: (row + 1) * 28, col * 28: (col + 1) * 28] = square
#
#     # Convert grayscale image to BGR for better visualization
#     combined_image = cv2.cvtColor(combined_image, cv2.COLOR_GRAY2BGR)
#
#     return combined_image




    # # Detect the largest quadrilateral (outer grid)
    # for cnt in contours:
    #     epsilon = 0.05 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #
    #     if len(approx) == 4:
    #         largest_contour = approx
    #         break
    # else:
    #     raise ValueError("Error: Could not detect a quadrilateral Sudoku grid.")

    # Reorder points for proper perspective transformation
    def reorder_points(pts):
        """
        Reorder the points to ensure they are in the correct top-left, top-right, bottom-right, bottom-left order.
        """
        pts = pts.reshape(4, 2)
        sum_pts = pts.sum(axis=1)
        diff_pts = np.diff(pts, axis=1)
        return np.array([pts[np.argmin(sum_pts)],  # Top-left
                         pts[np.argmin(diff_pts)],  # Top-right
                         pts[np.argmax(sum_pts)],  # Bottom-right
                         pts[np.argmax(diff_pts)],  # Bottom-left
                         ], dtype="float32")

    def save_squares_to_temp(squares, temp_dir):
        """
        Save 81 resized 28x28 squares into temporary memory with names corresponding to their positions.
        """
        saved_files = []
        for idx, square in enumerate(squares):
            # Generate a filename based on index position
            row = idx // 9
            col = idx % 9
            filename = f"{row}_{col}.png"
            file_path = os.path.join(temp_dir, filename)

            # Save the image to temporary directory
            cv2.imwrite(file_path, square)

            # Track the saved files for further processing
            saved_files.append(file_path)

        return saved_files

    def process_and_split_sudoku(img, largest_contour):
        """
        Warps the image perspective, splits it into 81 resized 28x28 squares, and saves them in temporary memory.
        """
        # Reorder points for perspective transformation
        ordered_pts = reorder_points(largest_contour)

        # Define the side length of the grid
        side = 450  # Pixels for the normalized grid
        dst = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype="float32")

        # Perform perspective transformation
        matrix = cv2.getPerspectiveTransform(ordered_pts, dst)
        warped = cv2.warpPerspective(img, matrix, (side, side))

        grid_size = 9
        square_size = side // grid_size

        # Divide the warped image into 81 squares and resize them to 28x28
        squares = []
        for row in range(grid_size):
            for col in range(grid_size):
                x_start = col * square_size
                y_start = row * square_size
                x_end = x_start + square_size
                y_end = y_start + square_size
                square = warped[y_start:y_end, x_start:x_end]
                # Resize each square to 28x28
                resized_square = cv2.resize(cv2.cvtColor(square, cv2.COLOR_BGR2GRAY), (28, 28))
                squares.append(resized_square)


        # Create a temporary directory to save all squares
        save_path = os.path.join(SAVE_DIR, f"{row}_{col}.png")
        cv2.imwrite(save_path, resized_square)
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_files = save_squares_to_temp(squares, temp_dir)

        return saved_files


def load_cnn_model():
    """
    Dynamically load the CNN model on each request.
    Replace with your own CNN model path or loading logic.
    """
    # Load your trained CNN model dynamically
    # Adjust 'path_to_model' with the correct path for your CNN model.
    model_path = "path_to_your_cnn_model"
    model = tf.keras.models.load_model(model_path)
    return model


def process_and_detect_numbers(image_paths):
    """
    Process a list of image paths with the CNN model and save only valid predictions.

    Args:
    - image_paths (list of str): List of image file paths to process (28x28 grayscale).

    Returns:
    - detected_files (list of str): List of valid processed file paths with detected numbers.
    """
    model = load_cnn_model()  # Dynamically load the CNN model
    detected_files = []
    failed_predictions = []

    for image_path in image_paths:
        try:
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Ensure it is 28x28
            if image.shape != (28, 28):
                raise ValueError(f"Unexpected image size: {image.shape}")

            # Normalize and reshape image for prediction
            image = image.astype(np.float32) / 255.0  # Scale pixel values between 0 and 1
            image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input

            # Run CNN prediction
            prediction = model.predict(image)
            predicted_number = np.argmax(prediction) + 1  # CNN outputs indices, convert to 1-9

            # Only save numbers if valid (between 1 and 9)
            if 1 <= predicted_number <= 9:
                # Extract row/column information from the filename
                filename = os.path.basename(image_path)
                row, col = map(int, filename.replace(".png", "").split("_"))
                save_filename = f"{predicted_number}_{row}_{col}.png"

                # Save the image to a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    save_path = os.path.join(temp_dir, save_filename)
                    cv2.imwrite(save_path, image * 255)  # Scale back to 0-255 range for saving
                    detected_files.append(save_path)

            else:
                failed_predictions.append(image_path)
        except Exception as e:
            # Log failed predictions
            failed_predictions.append(image_path)

    # Log edge cases for analysis
    if len(failed_predictions) == len(image_paths):
        raise ValueError("No numbers were detected across all predictions.")

    return detected_files


# Example usage
image_list = ["2_0.png", "3_2.png", "5_4.png"]  # Replace with paths to actual 28x28 images
try:
    result_files = process_and_detect_numbers(image_list)
    print(f"Detected numbers saved at these paths: {result_files}")
except Exception as e:
    print(f"Processing failed with error: {e}")


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles file upload from the user and returns the generated image preview of resized 81 squares.
    """
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("error.html", message="No file uploaded."), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = os.path.join(temp_dir, filename)
                file.save(temp_path)

                try:
                    combined_image = warp_and_split_sudoku(temp_path, temp_dir)
                except ValueError as e:
                    return render_template("error.html", message=str(e)), 400

                # Encode image to bytes for sending to frontend
                _, buffer = cv2.imencode(".jpg", combined_image)
                preview_bytes = BytesIO(buffer.tobytes())

                return send_file(preview_bytes, mimetype="image/jpeg")
        else:
            return render_template("error.html", message="Invalid file type. Only PNG, JPG, and JPEG are allowed."), 400
    else:
        return render_template("index.html")


if __name__ == "__main__":
    # Heroku Port Dynamic Handling
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
#git remote add heroku https://git.heroku.com/suduko-solver-app.git
#git push main
