#!/usr/bin/env python3
"""Capture synchronized image pairs from two Intel RealSense cameras (e.g. D435i)
in the folder layout expected by MC-Calib:

    <out>/Cam_001/000000.png  <out>/Cam_002/000000.png
    <out>/Cam_001/000001.png  <out>/Cam_002/000001.png
    ...

MC-Calib matches frames across cameras by their sorted index, so both folders
always receive the same number of images with identical names.

Controls (in the preview window):
    SPACE  save the current pair
    a      toggle auto-capture (saves a pair every --auto-interval seconds,
           only when the board is detected in both cameras)
    q/ESC  quit

The preview overlays the ChArUco corners detected with the dictionary given by
--dict (must equal aruco_dictionary in the MC-Calib config). If no corners show
up while the board is clearly visible, the dictionary / board size / layout do
not match your board and MC-Calib will not detect it either.
"""

import argparse
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/D435i_stereo/images",
                   help="output root folder (MC-Calib root_path)")
    p.add_argument("--serials", nargs=2, default=None, metavar=("CAM1", "CAM2"),
                   help="serial numbers of the cameras; first one becomes Cam_001 "
                        "(the reference camera). Default: two first devices, sorted")
    p.add_argument("--stream", choices=["color", "ir"], default="color",
                   help="color sensor or left infrared imager (depth frame)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hw-sync", action="store_true",
                   help="enable hardware sync (Cam_001 master, Cam_002 slave); "
                        "requires a sync cable between the cameras")
    p.add_argument("--prefix", default="Cam_", help="MC-Calib cam_prefix")
    p.add_argument("--squares-x", type=int, default=0,
                   help="board squares in X, for the detection preview (0 = markers only)")
    p.add_argument("--squares-y", type=int, default=0,
                   help="board squares in Y, for the detection preview")
    p.add_argument("--dict", default="DICT_4X4_1000",
                   help="ArUco dictionary of the board; must match aruco_dictionary "
                        "in the MC-Calib config (e.g. DICT_4X4_50, DICT_6X6_1000)")
    p.add_argument("--marker-ratio", type=float, default=0.75,
                   help="marker side / square side of your board (preview only)")
    p.add_argument("--auto-interval", type=float, default=1.0,
                   help="seconds between auto-captures")
    return p.parse_args()


class BoardDetector:
    """ChArUco detection for preview, compatible with old and new OpenCV APIs."""

    def __init__(self, squares_x, squares_y, marker_ratio, dict_name="DICT_6X6_1000"):
        aruco = cv2.aruco
        name = dict_name.upper()
        if not name.startswith("DICT_"):
            name = "DICT_" + name
        if not hasattr(aruco, name):
            raise SystemExit(f"Unknown ArUco dictionary {dict_name}")
        self.dict = aruco.getPredefinedDictionary(getattr(aruco, name))
        self.board = None
        self.new_api = hasattr(aruco, "CharucoDetector")
        self.charuco_dets = []  # (layout name, detector) for new OpenCV API
        if squares_x > 0 and squares_y > 0:
            if self.new_api:
                # Boards with an even number of rows are laid out differently
                # in OpenCV < 4.6 ("legacy") and >= 4.6. Try both and report
                # which one matches the printed board.
                for legacy in (False, True):
                    board = aruco.CharucoBoard((squares_x, squares_y), 1.0,
                                               marker_ratio, self.dict)
                    if legacy:
                        if not hasattr(board, "setLegacyPattern") or squares_y % 2:
                            continue
                        board.setLegacyPattern(True)
                    self.charuco_dets.append(
                        ("legacy(<4.6)" if legacy else "opencv>=4.6",
                         aruco.CharucoDetector(board)))
                self.board = self.charuco_dets[0][1].getBoard()
            else:
                self.board = aruco.CharucoBoard_create(squares_x, squares_y, 1.0,
                                                       marker_ratio, self.dict)
        self.layout = ""
        if self.new_api:
            self.aruco_det = aruco.ArucoDetector(self.dict,
                                                 aruco.DetectorParameters())

    def detect(self, gray):
        """Return (n_markers, charuco_corners or None, marker_corners, ids)."""
        aruco = cv2.aruco
        if self.new_api:
            if self.charuco_dets:
                best = None
                for name, d in self.charuco_dets:
                    ch_c, _, m_c, m_ids = d.detectBoard(gray)
                    n_c = 0 if ch_c is None else len(ch_c)
                    if best is None or n_c > best[0]:
                        best = (n_c, name, ch_c, m_c, m_ids)
                n_c, self.layout, ch_c, m_c, m_ids = best
                if n_c == 0:
                    self.layout = ""
                return (0 if m_ids is None else len(m_ids)), ch_c, m_c, m_ids
            m_c, m_ids, _ = self.aruco_det.detectMarkers(gray)
            return (0 if m_ids is None else len(m_ids)), None, m_c, m_ids
        m_c, m_ids, _ = aruco.detectMarkers(gray, self.dict)
        n = 0 if m_ids is None else len(m_ids)
        ch_c = None
        if self.board is not None and n > 0:
            _, ch_c, _ = aruco.interpolateCornersCharuco(m_c, m_ids, gray, self.board)
        return n, ch_c, m_c, m_ids


def start_camera(serial, args, sync_mode):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    if args.stream == "color":
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    else:
        cfg.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    profile = pipe.start(cfg)
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    if args.stream == "ir" and depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)  # no IR dot pattern
    if sync_mode is not None and depth_sensor.supports(rs.option.inter_cam_sync_mode):
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
    if args.stream == "color":
        sp = profile.get_stream(rs.stream.color)
    else:
        sp = profile.get_stream(rs.stream.infrared, 1)
    return pipe, sp.as_video_stream_profile().get_intrinsics()


def get_image(pipe, stream):
    frames = pipe.wait_for_frames()
    f = frames.get_color_frame() if stream == "color" else frames.get_infrared_frame(1)
    img = np.asanyarray(f.get_data()).copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img, f.get_timestamp()


def write_factory_intrinsics(path, serials, intrinsics, stream):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", len(serials))
    fs.write("stream", stream)
    for i, (s, intr) in enumerate(zip(serials, intrinsics)):
        fs.startWriteStruct(f"camera_{i}", cv2.FileNode_MAP)
        fs.write("serial", s)
        fs.write("camera_matrix", np.array([[intr.fx, 0, intr.ppx],
                                            [0, intr.fy, intr.ppy],
                                            [0, 0, 1]], dtype=np.float64))
        fs.write("distortion_vector", np.array([intr.coeffs], dtype=np.float64))
        fs.write("distortion_model_rs", str(intr.model))
        fs.write("img_width", intr.width)
        fs.write("img_height", intr.height)
        fs.endWriteStruct()
    fs.release()


def draw(img, det, label, layout=""):
    n_markers, ch_c, m_c, m_ids = det
    vis = img.copy()
    if n_markers:
        cv2.aruco.drawDetectedMarkers(vis, m_c, m_ids)
    n_corners = 0 if ch_c is None else len(ch_c)
    if n_corners:
        for c in ch_c.reshape(-1, 2):
            cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 255, 0), -1)
    cv2.putText(vis, f"{label}  markers:{n_markers}  corners:{n_corners}  {layout}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return vis, n_markers, n_corners


def main():
    args = parse_args()

    serials = args.serials
    if serials is None:
        found = sorted(d.get_info(rs.camera_info.serial_number) for d in rs.context().devices)
        if len(found) < 2:
            raise SystemExit(f"Need 2 RealSense cameras, found {len(found)}: {found}")
        serials = found[:2]
    print(f"Cam_001 (reference) = {serials[0]}\nCam_002             = {serials[1]}")

    cam_dirs = [os.path.join(args.out, f"{args.prefix}{i + 1:03d}") for i in range(2)]
    for d in cam_dirs:
        os.makedirs(d, exist_ok=True)
    # continue numbering if the folders already contain images
    existing = [len([f for f in os.listdir(d) if f.endswith(".png")]) for d in cam_dirs]
    if existing[0] != existing[1]:
        raise SystemExit(f"Cam folders contain different numbers of images {existing}; "
                         "clean them first (MC-Calib pairs frames by index).")
    idx = existing[0]

    sync = (1, 2) if args.hw_sync else (None, None)
    pipes, intrinsics = [], []
    for s, m in zip(serials, sync):
        p, intr = start_camera(s, args, m)
        pipes.append(p)
        intrinsics.append(intr)

    parent = os.path.dirname(os.path.abspath(args.out))
    write_factory_intrinsics(os.path.join(parent, "realsense_factory_intrinsics.yml"),
                             serials, intrinsics, args.stream)
    with open(os.path.join(parent, "camera_mapping.txt"), "w") as f:
        for i, s in enumerate(serials):
            f.write(f"{args.prefix}{i + 1:03d} {s}\n")

    detector = BoardDetector(args.squares_x, args.squares_y, args.marker_ratio, args.dict)
    max_dt = 1000.0 / args.fps  # ms
    auto, last_auto = False, 0.0
    print("SPACE: save pair | a: toggle auto-capture | q: quit")
    try:
        # let auto-exposure settle
        for _ in range(30):
            for p in pipes:
                p.wait_for_frames()
        while True:
            (img1, t1), (img2, t2) = (get_image(p, args.stream) for p in pipes)
            d1 = detector.detect(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
            v1, m1, c1 = draw(img1, d1, "Cam_001", detector.layout)
            d2 = detector.detect(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
            v2, m2, c2 = draw(img2, d2, "Cam_002", detector.layout)
            vis = np.hstack([v1, v2])
            status = f"saved: {idx}   auto: {'ON' if auto else 'off'}   dt: {abs(t1 - t2):.1f} ms"
            cv2.putText(vis, status, (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)
            scale = min(1.0, 1600.0 / vis.shape[1])
            cv2.imshow("RealSense pair capture", cv2.resize(vis, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF

            seen_both = (c1 if detector.board is not None else m1) > 0 and \
                        (c2 if detector.board is not None else m2) > 0
            save = key == ord(" ")
            if auto and seen_both and time.time() - last_auto > args.auto_interval:
                save, last_auto = True, time.time()
            if save:
                if args.hw_sync and abs(t1 - t2) > max_dt:
                    print(f"  skipped: timestamps differ by {abs(t1 - t2):.1f} ms")
                else:
                    name = f"{idx:06d}.png"
                    cv2.imwrite(os.path.join(cam_dirs[0], name), img1)
                    cv2.imwrite(os.path.join(cam_dirs[1], name), img2)
                    print(f"  saved pair {name}  (corners: {c1} / {c2})")
                    idx += 1
            if key == ord("a"):
                auto = not auto
            if key in (ord("q"), 27):
                break
    finally:
        for p in pipes:
            p.stop()
        cv2.destroyAllWindows()
    print(f"{idx} pairs in {args.out}")


if __name__ == "__main__":
    main()
