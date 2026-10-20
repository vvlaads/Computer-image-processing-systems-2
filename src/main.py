import cv2
from pathlib import Path
import numpy as np

INPUT_DIR = Path("../input")
OUTPUT_DIR = Path("../output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 10
MIN_RANSAC_INLIERS = 10
MIN_INLIER_RATIO = 0.4
MIN_OBJECT_AREA = 25


# Find keypoints and descriptors in a frame
def find_keypoints(sift, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)
    return keypoints, descriptors


# Match descriptors and filter them using Lowe's ratio test
def find_matches(matcher, descriptors_first, descriptors_second):
    matches = matcher.knnMatch(
        descriptors_first,
        descriptors_second,
        k=2
    )

    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            first_match, second_match = pair

            if first_match.distance < RATIO_THRESHOLD * second_match.distance:
                good_matches.append(first_match)

    return good_matches


# Check whether the transformed polygon is plausible
def is_valid_polygon(polygon, width, height):
    if not np.isfinite(polygon).all():
        return False

    points = polygon.reshape(-1, 2)

    # A projected rectangle should remain a convex quadrilateral
    if not cv2.isContourConvex(points.astype(np.float32)):
        return False

    area = abs(cv2.contourArea(points.astype(np.float32)))

    # Reject degenerate or implausibly large polygons
    if area < MIN_OBJECT_AREA or area > width * height * 4:
        return False

    return True


# Open video file
while True:
    video_name = input("Enter video name: ")
    video_path = INPUT_DIR / video_name

    if video_path.is_file():
        break

    print(f"Not found {video_path.name}. Try again.")

object_name = video_path.stem
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    raise SystemExit("Could not open video.")

# First frame
success, first_frame = video.read()

if not success:
    video.release()
    raise SystemExit("Could not read the first frame.")

height, width = first_frame.shape[:2]

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0

output_path = OUTPUT_DIR / f"{object_name}.mp4"

writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter.fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    video.release()
    raise SystemExit("Could not create output video.")

# Create SIFT and matcher
sift = cv2.SIFT.create()
matcher = cv2.BFMatcher(cv2.NORM_L2)

# Reference frame
reference_frame = first_frame.copy()
reference_keypoints, reference_descriptors = find_keypoints(sift, reference_frame)

if reference_descriptors is None or len(reference_keypoints) < 4:
    video.release()
    writer.release()
    raise SystemExit("Not enough keypoints in the first frame.")

# Initial object corners
tracked_corners = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
]).reshape(-1, 1, 2)

writer.write(first_frame)

processed_frames = 1
tracked_frames = 1
failed_frames = 0

while True:
    success, frame = video.read()

    if not success:
        break

    keypoints, descriptors = find_keypoints(sift, frame)
    tracked = False

    if descriptors is not None and reference_descriptors is not None:
        good_matches = find_matches(
            matcher,
            reference_descriptors,
            descriptors
        )

        if len(good_matches) >= MIN_GOOD_MATCHES:
            reference_points = np.float32([
                reference_keypoints[m.queryIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            frame_points = np.float32([
                keypoints[m.trainIdx].pt
                for m in good_matches
            ]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(
                reference_points,
                frame_points,
                cv2.RANSAC,
                RANSAC_THRESHOLD
            )

            if H is not None and mask is not None:
                inlier_count = int(mask.sum())
                inlier_ratio = inlier_count / len(good_matches)

                if inlier_count >= MIN_RANSAC_INLIERS and inlier_ratio >= MIN_INLIER_RATIO:
                    candidate_corners = cv2.perspectiveTransform(
                        tracked_corners,
                        H
                    )

                    # Update the reference only after a successful match
                    if is_valid_polygon(candidate_corners, width, height):
                        tracked_corners = candidate_corners

                        reference_frame = frame.copy()
                        reference_keypoints = keypoints
                        reference_descriptors = descriptors

                        tracked = True

    if tracked:
        tracked_frames += 1

        polygon = np.int32(tracked_corners)

        cv2.polylines(
            frame,
            [polygon],
            True,
            (0, 255, 0),
            3
        )

        # Get the top-left corner of the object
        x, y = polygon[0, 0]

        # Place the label above the object
        text_y = max(y - 10, 25)

        # Draw green text over the outline
        cv2.putText(
            frame,
            object_name,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    else:
        failed_frames += 1

        cv2.putText(
            frame,
            "Failed",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    writer.write(frame)
    processed_frames += 1

    cv2.imshow("Object Tracking", frame)

    # Stop processing
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
writer.release()
cv2.destroyAllWindows()

print(f"Output saved to: {output_path}")

# Statistics
tracking_rate = tracked_frames / processed_frames * 100
failure_rate = failed_frames / processed_frames * 100

print(f"Processed frames: {processed_frames}")
print(
    f"Tracked frames: {tracked_frames} "
    f"({tracking_rate:.2f}%)"
)
print(
    f"Failed frames: {failed_frames} "
    f"({failure_rate:.2f}%)"
)
