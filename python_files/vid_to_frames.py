import cv2
import os
import argparse


"""usage:
python vid_to_frames.py input.mp4 -o frames -n 1 -f jpg
"""

def extract_frames(video_path, output_dir, every_n_frames=1, image_format="jpg"):
    """
    Extract frames from an mp4 video and save them as images.

    Args:
        video_path (str): Path to the input .mp4 file.
        output_dir (str): Directory to save extracted frames.
        every_n_frames (int): Save every Nth frame (1 = save all frames).
        image_format (str): Output image format ("jpg" or "png").
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f} | Total frames: {total_frames}")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            filename = os.path.join(output_dir, f"frame_{saved_idx:06d}.{image_format}")
            cv2.imwrite(filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_idx} frames to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an .mp4 video.")
    parser.add_argument("video_path", help="Path to the input .mp4 file")
    parser.add_argument("-o", "--output_dir", default="frames", help="Directory to save frames (default: frames)")
    parser.add_argument("-n", "--every_n_frames", type=int, default=1,
                         help="Save every Nth frame (default: 1, saves all frames)")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png"],
                         help="Output image format (default: jpg)")

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.every_n_frames, args.format)