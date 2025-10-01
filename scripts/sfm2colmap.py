import json, os, shutil
from os.path import join, basename
from tqdm import tqdm
import numpy as np
import argparse


def make_dir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def write_points3D(sfm_data, output : str = "sparse/0"):
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    points3D = sfm_data["structure"]
    with open(join(output, "points3D.txt"), "w") as f:
        for p in tqdm(points3D):
            f.write(
                f"{p['landmarkId']} {p['X'][0]} {p['X'][1]} {p['X'][2]} {p['color'][0]} {p['color'][1]} {p['color'][2]} 0.0 "
            )
            for v in p["observations"]:
                f.write(f"{v['observationId']} {v['featureId']} ")
            f.write("\n")

    with open(join(output, "pointcloud.obj"), "w") as f:
        for p in tqdm(points3D):
            f.write(f"v {p['X'][0]} {p['X'][1]} {p['X'][2]}\n")


def write_cameras(sfm_data, output : str = "sparse/0", camera_model : str = "PINHOLE"):
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    intrinsics = dict()
    for intrinsic in sfm_data["intrinsics"]:
        intrinsics[intrinsic["intrinsicId"]] = intrinsic
    cameras = sfm_data["views"]
    with open(join(output, "cameras.txt"), "w") as f:
        for c in tqdm(cameras):
            ins = intrinsics[c["intrinsicId"]]
            
            focal_length_mm = float(ins['focalLength'])
            sensor_width_mm = float(ins['sensorWidth'])
            image_width_px = float(ins['width'])
            
            focal_length_px = (focal_length_mm / sensor_width_mm) * image_width_px
            
            f.write(
                f"{c['viewId']} {camera_model} {ins['width']} {ins['height']} {focal_length_px} {focal_length_px} "
                + f"{ins['principalPoint'][0]} {ins['principalPoint'][1]}\n"
            )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_images(sfm_data, output : str = "sparse/0"):
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # POINTS2D[] as (X, Y, POINT3D_ID)
    cameras = sfm_data["views"]
    poses = {pose["poseId"]: pose["pose"]["transform"] for pose in sfm_data["poses"]}
    features = dict()
    for p in sfm_data["structure"]:
        for v in p["observations"]:
            vid = v["observationId"]
            if vid not in features:
                features[vid] = dict()
            features[vid][v["featureId"]] = (v["x"][0], v["x"][1], p["landmarkId"])

    with open(join(output, "images.txt"), "w") as f:
        for c in tqdm(cameras):
            poseId = c["poseId"]
            if poseId not in poses:
                continue

            r = np.asarray(poses[poseId]["rotation"], dtype=float).reshape(3, 3)
            t = np.asarray(poses[poseId]["center"], dtype=float)

            r = r.T
            t = -r @ t
            qvec = rotmat2qvec(r)
            f.write(
                f"{c['viewId']} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t[0]} {t[1]} {t[2]} {c['viewId']} {basename(c['path'])}\n"
            )

            sub_features = features[c["viewId"]]
            for feature in sub_features.values():
                f.write(f"{feature[0]} {feature[1]} {feature[2]} ")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Convert Meshroom sfm.json to COLMAP format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the Meshroom sfm.json file.")
    parser.add_argument("--output_dir", type=str, required=False, help="Path to the output directory for COLMAP files (default: sparse/0).")
    parser.add_argument("--camera_model", type=str, help="Specifies the COLMAP camera model to use. Available models: PINHOLE, SIMPLE_RADIAL, RADIAL etc. (default: PINHOLE)")
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        sfm_data = json.load(f)
    print(
        f"# of images: {len(sfm_data['poses'])}\n# of 3D points: {len(sfm_data['structure'])}"
    )
    make_dir(args.output_dir)
    write_points3D(sfm_data, args.output_dir)
    write_cameras(sfm_data, args.output_dir, args.camera_model)
    write_images(sfm_data, args.output_dir)


if __name__ == "__main__":
    main()
