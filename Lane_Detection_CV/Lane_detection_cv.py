import cv2
import numpy as np


def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)


def get_line_markings(frame):
    bgr2hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    _, sxbinary = threshold(bgr2hls[:, :, 1], thresh=(100, 255))
    _, s_binary = threshold(bgr2hls[:, :, 2], thresh=(40, 255))
    _, r_thresh = threshold(frame[:, :, 2], thresh=(90, 255))
    rs_binary = cv2.bitwise_and(s_binary, r_thresh)
    return cv2.cvtColor(rs_binary, cv2.COLOR_GRAY2BGR)


def canny(frame):
    highlighted_frame = get_line_markings(frame)
    kernel = 17
    blur_frame = cv2.GaussianBlur(highlighted_frame, (kernel, kernel), 1)
    return cv2.Canny(blur_frame, 155, 200)


def region_of_interest(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [vertices], 255)
    return cv2.bitwise_and(frame, mask)


def hough_lines(frame):
    return cv2.HoughLinesP(frame, 4, np.pi / 180, 35, minLineLength=80, maxLineGap=100)


class LaneTracker:
    def __init__(self):
        self.last_left_line = None
        self.last_right_line = None

    def update(self, left_line, right_line):
        if left_line is not None:
            self.last_left_line = left_line
        if right_line is not None:
            self.last_right_line = right_line

    def get_lines(self):
        return [self.last_left_line, self.last_right_line]


def make_points(frame, line):
    height, width = frame.shape[:2]
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 3 / 5)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, lines):
    left_lines, right_lines = [], []
    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    left_line = right_line = None
    if len(left_lines) > 0:
        left_avg = np.average(left_lines, axis=0)
        left_line = make_points(frame, left_avg)
    if len(right_lines) > 0:
        right_avg = np.average(right_lines, axis=0)
        right_line = make_points(frame, right_avg)

    return left_line, right_line


def display_lines(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                cv2.line(frame, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 3)
    return cv2.addWeighted(frame, 0.8, line_image, 1, 0)


def lane_detection_process(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    roi_vertices = np.array([[(0, frame_height), (0, 480), (390, 300), (588, 300), (frame_width, frame_height)]],
                            dtype=np.int32)

    lane_tracker = LaneTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            canny_output = canny(frame)
            roi_output = region_of_interest(canny_output, roi_vertices)
            lines = hough_lines(roi_output)

            left_line, right_line = average_slope_intercept(frame, lines)
            lane_tracker.update(left_line, right_line)

            current_lines = lane_tracker.get_lines()
            line_image = display_lines(frame, current_lines)

            out.write(line_image)
            cv2.imshow("Lane Detection", line_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Usage
lane_detection_process("Lane_video.mp4", "Lane_detected_optimized.mp4")