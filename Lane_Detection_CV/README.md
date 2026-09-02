**Markdown**

**# Computer Vision Lane Detection Pipeline**



**## Overview**

**This project implements a robust lane detection and tracking algorithm using OpenCV and classical Computer Vision techniques. It processes video input frame-by-frame, highlights lane lines using custom color thresholds, isolates edges using Canny edge detection, applies a Region of Interest (ROI) mask, and uses Hough Transform with temporal line tracking to maintain smooth lane boundaries.**



**## Key Features**

**\* \*\*Color Space Conversion:\*\* Converts BGR to HLS color space to isolate yellow and white lane markings under varying lighting conditions.**

**\* \*\*Canny Edge \& Gaussian Blur:\*\* Applies a 17x17 Gaussian blur to remove noise followed by Canny edge extraction.**

**\* \*\*Region of Interest (ROI) Masking:\*\* Dynamic polygon masking to filter out off-road scenery and focus strictly on the vehicle's driving lane.**

**\* \*\*Hough Transform \& Line Averaging:\*\* Detects line segments, separates them into left/right lanes by slope, and calculates averaged lane boundaries.**

**\* \*\*Temporal Line Tracking (`LaneTracker`):\*\* Keeps track of previous valid lane coordinates to prevent flickering when lines are briefly obscured or missing.**



**## Tech Stack**

**\* Python 3.x**

**\* OpenCV (`cv2`)**

**\* NumPy**



**## File Structure**

**\* `lane\_detection.py` - Core lane detection script.**

**\* `Lane\_video.mp4` - Input test video.**

**\* `Lane\_detected\_optimized.mp4` - Output video with drawn lane overlays.**



**## How to Run**

**```bash**

**python lane\_detection.py**


