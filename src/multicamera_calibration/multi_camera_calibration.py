"""Multi-camera calibration module."""

import os
import yaml
import shutil
import pycolmap
import numpy as np

from pathlib import Path


class MultiCameraCalibration:
    """Class for multi-camera calibration."""

    def __init__(self):
        """Initialize the MultiCameraCalibration object."""
        self.calib = None
        self.rig = None
        self.db_path = None
        self.image_path = None
        self.output_path = None
        self.image_to_id_dict = {}  # Mapping from image names to their corresponding IDs in the database
        self.id_to_image_dict = {}  # Mapping from database IDs to their corresponding image names

    def read_config(self, calib_file: str, rig_file: str):
        """
        Read calibration and rig configuration files.

        Args:
            calib_file: Path to the calibration YAML file.
            rig_file: Path to the rig configuration YAML file.

        Raises:
            FileNotFoundError: If either file does not exist.
            yaml.YAMLError: If there's an error parsing the YAML files.
        """
        # Read calibration file
        try:
            with open(calib_file, 'r') as f:
                self.calib = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing calibration file: {e}")

        # Read rig file
        try:
            with open(rig_file, 'r') as f:
                self.rig = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Rig file not found: {rig_file}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing rig file: {e}")

    def database_initialization(self, db_path: str, image_path: str):

        if os.path.exists(db_path):
            os.remove(db_path)

        self.db_path = Path(db_path)
        self.image_path = Path(image_path)
        cam0_image_path = Path(image_path) / "rig1" / "cam0"

        db = pycolmap.Database.open(db_path)
        cam0 = self.calib.get("cam0")
        cam1 = self.calib.get("cam1")
        frames = os.listdir(cam0_image_path)

        # Create cameras
        camera0 = pycolmap.Camera(cam0)
        camera1 = pycolmap.Camera(cam1)
        db.write_camera(camera0)
        db.write_camera(camera1)

        # Create rig
        colmap_rig = pycolmap.Rig(
            {
                "rig_id": 1,
            }
        )   
        sensor1 = pycolmap.sensor_t(
            {
                "type": pycolmap.SensorType.CAMERA, 
                "id": 1,  # Use camera_id 1 (first camera)
             }
             )
        colmap_rig.add_ref_sensor(sensor1)

        sensor2 = pycolmap.sensor_t(
            {
                "type": pycolmap.SensorType.CAMERA, 
                "id": 2,  # Use camera_id 2 (second camera)
             }
             )

        transform = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(self.rig['cam1']['rotation']),
            translation=self.rig['cam1']['translation']
        )
        colmap_rig.add_sensor(sensor2, transform)
        db.write_rig(colmap_rig)

        # Create images and frames
        image_id = 1
        frame_id = 1
        for frame in frames:
            image_cam0 = pycolmap.Image(
                name=f"rig1/cam0/{frame}",
                points2D=np.empty((0, 2), dtype=np.float64),
                camera_id=1,
                image_id=image_id
            )
            self.image_to_id_dict[frame] = image_id
            self.id_to_image_dict[image_id] = frame
            image_id += 1
            db.write_image(image_cam0, use_image_id=False)

            colmap_frame = pycolmap.Frame(
                {
                    "frame_id": frame_id,
                    "rig_id": 1,
                    #"image_ids": [],
                }
                )

            frame_id += 1

            image_cam1 = pycolmap.Image(
                name=f"rig1/cam1/{frame}",
                points2D=np.empty((0, 2), dtype=np.float64),
                camera_id=2,
                image_id=image_id
            )
            self.image_to_id_dict[frame] = image_id
            self.id_to_image_dict[image_id] = frame
            image_id += 1
            db.write_image(image_cam1, use_image_id=False)

            colmap_frame.add_data_id(image_cam0.data_id)
            colmap_frame.add_data_id(image_cam1.data_id)
            db.write_frame(colmap_frame)

        return db

    def run_calibration(self, output_path: str, known_distance_cam0_cam1: float):

        self.output_path = Path(output_path)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        pycolmap.extract_features(self.db_path, self.image_path)
        pycolmap.match_exhaustive(self.db_path)
        maps = pycolmap.incremental_mapping(self.db_path, self.image_path, self.output_path)
        maps[0].write_text(self.output_path)

        #print(maps[0].cameras)
        #print(maps[0].images)
        #rig1 = maps[0].rig(rig_id=1)
        #print(dir(rig1))

        with open(self.output_path / "cameras.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("1 "):
                    cam0_params = list(map(float, line.split()[4:]))
                elif line.startswith("2 "):
                    cam1_params = list(map(float, line.split()[4:]))
        print("Camera 0 parameters:", cam0_params)
        print("Camera 1 parameters:", cam1_params)

        with open(self.output_path / "rigs.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("1 "):
                    rig_params = list(map(float, line.split()[7:]))
                    rotation = np.array(rig_params[:4])
                    translation = np.array(rig_params[4:])
                    scale_factor = np.linalg.norm(translation) / known_distance_cam0_cam1
                    translation /= scale_factor

        print("Rig parameters (cam1 relative to cam0):", "rotation:", rotation, "translation:", translation)
