
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

DESK_CAM_CFG = CameraCfg(
    prim_path="/sensor/desk_sensor",
    update_period=0.1,
    height=480,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=16.0,
        focus_distance=400.0,
        horizontal_aperture=24,
        clipping_range=(0.1, 1.0e5),
    ),
)

HAND_CAM_CFG = CameraCfg(
        prim_path="/sensor/hand_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(0, 0, 0, 0),
        ),
)