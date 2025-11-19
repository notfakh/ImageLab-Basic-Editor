# ImageLab-Basic-Editor
A simple OpenCV-based image processing tool that loads an image, resizes it, rotates it, and applies multiple Gaussian blur variations to demonstrate core image manipulation techniques.

## 📋 Project Overview

This project showcases fundamental image manipulation techniques using OpenCV (cv2). It performs a series of transformations on an input image including resizing, rotation, and multiple blur effects with varying kernel sizes and sigma values to demonstrate their visual impact.

## 🎯 What Does This Do?

- **Loads** any image file (JPG, PNG, etc.)
- **Resizes** image to 200x200 pixels
- **Rotates** image 90° clockwise
- **Applies** Gaussian blur with default settings
- **Experiments** with 4 different blur configurations:
  - Small kernel (3x3)
  - Large kernel (9x9)
  - High sigma value (σ=10)
  - Combined large kernel + high sigma (9x9, σ=15)
- **Saves** all processed images automatically

## 🔑 Key Features

- ✅ Interactive image display with OpenCV windows
- ✅ Automatic saving of all processed images
- ✅ Error handling for file loading
- ✅ Multiple blur parameter demonstrations
- ✅ Step-by-step visualization
- ✅ Comparison of blur effects
- ✅ Simple and educational code structure

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
An image file (JPG, PNG, etc.)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notfakh/opencv-image-processing.git
cd opencv-image-processing
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Add your image:
   - Place your image file in the project directory
   - Name it `image.jpg` (or update the path in the code)

### Usage

Run the script:
```bash
python image_processing.py
```

**Interactive Display:**
- Each transformation displays in a new window
- Press any key to proceed to the next transformation
- All processed images are automatically saved

## 📊 Image Transformations

### 1. Original Image
- Displays the loaded image
- No modifications applied

### 2. Resize (200x200)
```python
cv2.resize(img, (200, 200))
```
- Output: `resized.jpg`
- Fixed dimensions: 200x200 pixels
- Maintains aspect ratio (may distort if original ratio differs)

### 3. Rotate 90° Clockwise
```python
cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
```
- Output: `rotated.jpg`
- Applied to the resized image
- Orientation: 90° clockwise

### 4. Gaussian Blur Variations

#### a) Default Blur (5x5 kernel, σ=0)
```python
cv2.GaussianBlur(img, (5, 5), 0)
```
- Output: `blurred.jpg`
- Moderate blur effect
- σ=0 means auto-calculated based on kernel size

#### b) Small Blur (3x3 kernel, σ=0)
```python
cv2.GaussianBlur(img, (3, 3), 0)
```
- Output: `blur_small.jpg`
- Minimal blur effect
- Preserves more detail

#### c) Large Blur (9x9 kernel, σ=0)
```python
cv2.GaussianBlur(img, (9, 9), 0)
```
- Output: `blur_large.jpg`
- Stronger blur effect
- Larger smoothing area

#### d) High Sigma Blur (5x5 kernel, σ=10)
```python
cv2.GaussianBlur(img, (5, 5), 10)
```
- Output: `blur_high_sigma.jpg`
- Wider Gaussian distribution
- More aggressive smoothing

#### e) Combined Blur (9x9 kernel, σ=15)
```python
cv2.GaussianBlur(img, (9, 9), 15)
```
- Output: `blur_combined.jpg`
- Maximum blur effect
- Large kernel + high sigma

## 📈 Understanding Gaussian Blur Parameters

### Kernel Size (ksize)
- **Definition**: The size of the blur matrix (must be odd: 3, 5, 7, 9, etc.)
- **Small (3x3)**: Subtle blur, preserves edges
- **Medium (5x5)**: Balanced smoothing
- **Large (9x9+)**: Strong blur, removes fine details

### Sigma X (sigmaX)
- **Definition**: Standard deviation in X direction
- **0**: Auto-calculated from kernel size
- **Low (1-5)**: Gentle smoothing
- **High (10+)**: Aggressive smoothing

### Visual Impact:
```
Kernel Size ↑ = More Blur
Sigma Value ↑ = More Blur
Both ↑ = Maximum Blur
```

## 🎨 Output Files

After running the script, you'll have:

| File | Description | Size |
|------|-------------|------|
| `resized.jpg` | 200x200 resized image | 200x200 |
| `rotated.jpg` | 90° clockwise rotation | 200x200 |
| `blurred.jpg` | Default blur (5x5, σ=0) | Original size |
| `blur_small.jpg` | Minimal blur (3x3, σ=0) | Original size |
| `blur_large.jpg` | Strong blur (9x9, σ=0) | Original size |
| `blur_high_sigma.jpg` | High sigma blur (5x5, σ=10) | Original size |
| `blur_combined.jpg` | Maximum blur (9x9, σ=15) | Original size |

## 🛠️ Customization

### Change Image Path
```python
image_path = "path/to/your/image.jpg"
```

### Modify Resize Dimensions
```python
resized = cv2.resize(img, (400, 300))  # Width x Height
```

### Try Different Rotations
```python
# 90° counter-clockwise
rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 180°
rotated = cv2.rotate(img, cv2.ROTATE_180)

# Custom angle
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45° rotation
rotated = cv2.warpAffine(img, M, (w, h))
```

### Experiment with Blur Parameters
```python
# Extreme blur
extreme_blur = cv2.GaussianBlur(img, (15, 15), 20)

# Minimal blur
minimal_blur = cv2.GaussianBlur(img, (3, 3), 1)

# Asymmetric blur (different X and Y sigma)
bilateral_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=10, sigmaY=5)
```

### Batch Processing Multiple Images
```python
import glob

for image_path in glob.glob("images/*.jpg"):
    img = cv2.imread(image_path)
    # Apply transformations...
```

## 💡 Use Cases

- **Image Preprocessing**: Prepare images for machine learning
- **Noise Reduction**: Remove image noise with Gaussian blur
- **Data Augmentation**: Create training variations
- **Privacy Protection**: Blur sensitive information
- **Artistic Effects**: Create soft-focus photography effects
- **Computer Vision**: Standard preprocessing for CV pipelines

## 🔬 Extending the Project

Ideas for enhancement:

1. **Add More Filters**
   ```python
   # Median blur
   median = cv2.medianBlur(img, 5)
   
   # Bilateral filter (edge-preserving)
   bilateral = cv2.bilateralFilter(img, 9, 75, 75)
   
   # Motion blur
   kernel = np.zeros((5, 5))
   kernel[int((5-1)/2), :] = np.ones(5)
   motion_blur = cv2.filter2D(img, -1, kernel/5)
   ```

2. **Edge Detection**
   ```python
   # Canny edge detection
   edges = cv2.Canny(img, 100, 200)
   
   # Sobel edges
   sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
   ```

3. **Color Space Conversions**
   ```python
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   ```

4. **Image Enhancement**
   ```python
   # Histogram equalization
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   equalized = cv2.equalizeHist(gray)
   
   # Contrast adjustment
   alpha = 1.5  # Contrast
   beta = 30    # Brightness
   adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
   ```

5. **Interactive GUI**
   ```python
   # Add trackbars for real-time parameter adjustment
   cv2.createTrackbar('Kernel Size', 'window', 1, 10, callback)
   cv2.createTrackbar('Sigma', 'window', 0, 50, callback)
   ```

6. **Comparison Grid**
   ```python
   # Create side-by-side comparison
   import numpy as np
   comparison = np.hstack([original, blurred, rotated])
   cv2.imshow("Comparison", comparison)
   ```

## 📊 Performance Tips

- **Large Images**: Consider downscaling before processing
- **Batch Processing**: Use multiprocessing for multiple images
- **Memory**: Close windows after viewing to free memory
- **Speed**: Smaller kernels process faster
- **Quality vs Speed**: Balance blur intensity with performance needs

## 🤝 Contributing

Contributions welcome! Enhancement ideas:

- Add GUI with sliders for interactive parameter adjustment
- Implement real-time webcam processing
- Add more filter types (median, bilateral, etc.)
- Create before/after comparison view
- Add batch processing for folders
- Implement image sharpening techniques
- Add perspective transformation

## 👤 Author

**Fakhrul Sufian**
- GitHub: [@notfakh](https://github.com/notfakh)
- LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- OpenCV library for computer vision tools
- NumPy for array operations
- Python community for documentation and tutorials

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Gaussian Blur Explained](https://en.wikipedia.org/wiki/Gaussian_blur)
- [Image Processing Tutorials](https://opencv-python-tutroals.readthedocs.io/)
- [Computer Vision Fundamentals](https://www.pyimagesearch.com/)

## 🐛 Troubleshooting

**Issue: Image not loading**
- Check file path is correct
- Ensure image file exists
- Try absolute path: `C:/full/path/to/image.jpg`

**Issue: Windows not displaying**
- Check if running in IDE (may need special configuration)
- Try adding longer `cv2.waitKey(0)` duration
- Ensure display is available (not in headless environment)

**Issue: Saved images are black/corrupted**
- Check original image loads correctly
- Verify write permissions in directory
- Ensure sufficient disk space

**Issue: Kernel size error**
- Kernel size must be odd (3, 5, 7, 9, etc.)
- Both width and height must be positive odd numbers

## 📧 Contact

For questions, suggestions, or collaboration:
- Open an issue in this repository
- Email: fkhrlnasry@gmail.com
- Connect on LinkedIn

---

⭐ If this project helped you learn OpenCV image processing, please give it a star!

## 🎓 Learning Outcomes

After working through this project, you'll understand:
- Loading and displaying images with OpenCV
- Image resizing and aspect ratio considerations
- Image rotation techniques
- Gaussian blur principles and parameters
- Kernel size vs sigma effects
- Saving processed images
- Basic computer vision operations

**Perfect for:** Computer vision beginners, image processing students, and OpenCV l
