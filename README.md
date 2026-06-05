# Hands-on Planning Project

Autonomous exploration of an unknown environment using a **Kobuki TurtleBot 2** with an **Intel RealSense D435i** camera, running in [Stonefish](https://stonefish.readthedocs.io) simulation and deployable on the real robot.

The robot builds a map with **RTAB-Map** (visual SLAM), selects unexplored frontiers, plans collision-free paths with **Bidirectional RRT\***, and executes them using a **Dynamic Window Approach (DWA)** local planner.

**Team:** Haadi / Huy / Phu

---

## System Architecture

```
┌─────────────────────────── Stonefish Simulator ───────────────────────────┐
│   RGB image       Depth image       Odometry         IMU (NED)            │
└───────┬────────────────┬──────────────────┬────────────────┬──────────────┘
        │                │                  │                │
  ImageCropNode     ImageCropNode       EKF Node       ImuNedToEnu
  (color crop)     (depth/scan crop)  (robot_local.)   (NED→ENU)
        │                │                  ▲                │
        │          depthimage_to_           │                │
        │           laserscan              /turtlebot/odom   │
        │        → /turtlebot/fake_scan     └────────────────┘
        │                │                  /odometry/filtered
        └────────────────┴─────────────────────────┬──────────────────────
                                                    ▼
                                              RTAB-Map (Visual SLAM)
                                             RGB-D + fake scan + EKF odom
                                                    │
                                                  /map
                                                    │
                                         GlobalCostmap (grid_mapping)
                                          (inflate obstacles 0.2 m)
                                                    │
                                             /inflated_map
                                          ┌──────────┴──────────┐
                                     FrontierNode         PathPlannerNode
                                    (BFS + scoring)        (BiRRT* global)
                                          │                      │
                                   /frontier_goal  ──────►  DWA Service
                                                           (local velocity)
                                                                  │
                                                       /turtlebot/cmd_vel
```

---

## Packages

### Core Packages

| Package | Purpose |
|---------|---------|
| `online_motion_planning` | Main orchestration: frontier selection, global path planning (BiRRT\*), IMU conversion, arm retraction, launch files |
| `dwa_planner` | DWA local planner, exposed as a ROS 2 service |
| `dwa_interfaces` | Custom service definition (`ComputeVelocity.srv`) |
| `grid_mapping` | Subscribes to `/map`, inflates obstacles, publishes `/inflated_map` |
| `localization` | Custom differential-drive EKF (wheel encoders + IMU) — used on the real robot |
| `frontier_based_exploration` | Standalone simpler frontier node (legacy; the main pipeline uses `frontier_node` in `online_motion_planning`) |

### Sensor / Utility Packages

| Package | Purpose |
|---------|---------|
| `image_utils` | Crops RGB and depth images for RTAB-Map; synchronises RTAB-Map input streams |
| `scan_to_cloud2` | Converts `LaserScan` → `PointCloud2` |
| `depthimage_to_laserscan` | Converts a cropped depth image row → fake `LaserScan` (`/turtlebot/fake_scan`) |

### Third-Party Packages (vendored)

| Package | Purpose |
|---------|---------|
| `rtabmap` | RTAB-Map SLAM library (C++ core) |
| `rtabmap_ros` | ROS 2 wrappers for RTAB-Map |
| `turtlebot_simulation` | Stonefish scenario files + simulation launch files |
| `kobuki_description`, `swiftpro_description`, `turtlebot_description` | URDF models and meshes |

---

## How It Works

### 1. Sensor Pipeline

The **RealSense D435i** outputs full-resolution RGB and depth. `image_crop_node` produces two variants:

- **`image_cropped`** — bottom fraction removed (cuts out the floor) → fed to RTAB-Map
- **`image_scan_cropped`** — top and bottom removed (cuts the robot arm and floor) → fed to `depthimage_to_laserscan`

`depthimage_to_laserscan` converts that narrow depth slice into a `LaserScan` on `/turtlebot/fake_scan`, giving RTAB-Map a 2D range input without a dedicated LiDAR.

Stonefish publishes the IMU in NED (North-East-Down) convention. `imu_ned_to_enu` negates yaw and ω_z before the data reaches the EKF. The `robot_localization` EKF node fuses wheel odometry (`/turtlebot/odom`) and the converted IMU heading to produce `/odometry/filtered`.

### 2. Mapping

**RTAB-Map** runs visual-inertial SLAM using cropped RGB-D + fake scan + EKF odometry. It publishes:
- `/map` (`OccupancyGrid`) — live 2D occupancy map
- TF transform `world_enu → turtlebot/base_footprint`

The **global costmap** node (`occupancy_grid_original`) takes `/map` and re-publishes `/inflated_map`, expanding every occupied cell by `inflation_radius` (default 0.2 m) so planners maintain clearance from walls automatically.

### 3. Exploration Loop

```
FrontierNode detects unexplored boundary cells on /map
  → clusters with connected-components (OpenCV)
  → scores each cluster (size, distance, angle)
  → BFS reachability check on /inflated_map
  → publishes best cluster to /frontier_goal

PathPlannerNode receives /frontier_goal
  → builds binary obstacle map from /inflated_map
  → runs Bidirectional RRT* to find a collision-free path
  → smooths path (greedy shortcutting)
  → feeds waypoints one-by-one to DWA service

DWA service (/dwa/compute_velocity)
  → receives (goal_x, goal_y)
  → samples velocity candidates in dynamic window
  → scores trajectories: heading, distance, obstacle clearance, speed
  → returns best (linear_x, angular_z)

PathPlannerNode publishes cmd_vel to robot
  → on waypoint reached → request next waypoint
  → on goal reached → notify FrontierNode → pick new frontier

FrontierNode finds no reachable frontiers
  → publishes /frontier/exploration_complete → robot halts
```

### 4. Bidirectional RRT\*

`BIRRT_STAR` extends `RRT_STAR`. Both share:

- `rand_conf()` — random free-cell sampling with configurable goal bias
- `nearest_vertex()` — nearest-neighbour search
- `new_conf()` — one-step tree extension toward a target
- `rewire_qnear_to_qnew()` / `rewire_qnew_from_qnear()` — RRT\* cost rewiring
- `is_segment_free_bisection()` — recursive bisection collision check
- `smoothing()` — greedy path shortcutting

`BIRRT_STAR` overrides only `sample()`: it grows two trees simultaneously — T_a from the start and T_b from the goal — and attempts to bridge them via `_try_connect()` at each iteration. When they connect, `_build_unified()` merges the two sub-paths into one sequential path. Because both ends grow toward the middle, paths are found roughly twice as fast in cluttered maps compared to single-tree RRT\*.

---

## Prerequisites

### System Dependencies

```bash
# ROS 2 Humble or later
sudo apt install \
    ros-humble-robot-localization \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-laser-geometry \
    xterm
```

You also need [Stonefish](https://stonefish.readthedocs.io) and `stonefish_ros2` installed and sourced.

### Python Dependencies

```bash
pip install numpy opencv-python
```

---

## Building

```bash
cd <workspace_root>    # the directory containing src/
colcon build --symlink-install
source install/setup.bash
```

To build only the custom packages (faster on a clean build):

```bash
colcon build --symlink-install --packages-select \
    dwa_interfaces dwa_planner grid_mapping image_utils localization \
    online_motion_planning scan_to_cloud2 turtlebot_simulation
```

---

## Running

### Simulation

One command starts the full stack — Stonefish + RTAB-Map + all planning nodes:

```bash
source install/setup.bash
ros2 launch online_motion_planning exploration_sim.launch.py
```

The launch file staggers node startup automatically:

| Delay | What starts |
|-------|------------|
| 0 s | Stonefish (`turtlebot_hoi_circuit2`), IMU converter, EKF node |
| 5 s | Image crop node, arm retract node, RViz |
| 8 s | RTAB-Map (opens in its own xterm window) |
| 5 s | Global costmap, DWA local costmap, DWA service, frontier node, path planner |

### Real Robot

#### Step 1 — Hardware Drivers (before the launch file)

These must be running **before** `turtlebot_real_test.launch.py` is launched. Start each in its own terminal:

```bash
# Terminal 1 — Kobuki base (publishes /turtlebot/odom, /turtlebot/joint_states, cmd_vel)
ros2 launch kobuki_ros kobuki.launch.py

# Terminal 2 — RealSense D435i camera driver
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=turtlebot \
    camera_name:=camera \
    enable_depth:=true \
    enable_color:=true \
    align_depth.enable:=true

# Terminal 3 — EKF-based odometry (robot_localization, fuses wheel odom)
#   This is already included in turtlebot_real_test.launch.py — no need to
#   launch it separately unless you want to test localisation in isolation.
```

> The RealSense driver publishes color on `/turtlebot/camera/color/image_compressed` and depth on `/turtlebot/camera/depth/image_rect_raw`. These are remapped inside the launch file automatically.

#### Step 2 — Launch the Planning Stack

```bash
source install/setup.bash
ros2 launch online_motion_planning turtlebot_real_test.launch.py
```

The launch sequence on the real robot:

| Delay | What starts |
|-------|------------|
| 0 s | EKF node (`robot_localization`), RTAB-Map (in its own xterm window) |
| 2 s | Image crop node, `depthimage_to_laserscan`, scan-based global map (`scan_map_global`), RViz |
| 8 s | DWA local costmap, DWA service, frontier node, path planner |

#### Key Differences from Simulation

On the real robot the pipeline is meaningfully different in several places:

**Map source for planning** — In simulation, planning uses the RTAB-Map `/map` topic inflated by `grid_mapping`. On the real robot, `scan_map_global` builds a persistent occupancy grid directly from the fake LiDAR scan (depth image → `depthimage_to_laserscan`). This produces `/map_scan` and `/inflated_map_scan`, which frontier detection and path planning consume instead:

```
Sim:   RTAB-Map → /map          → grid_mapping → /inflated_map
Real:  fake scan → scan_map_global → /map_scan  → /inflated_map_scan
```

**No IMU frame conversion** — The real robot's IMU already publishes in ENU; no `imu_ned_to_enu` node is needed.

**No arm retract node** — `is_sim: false` means `arm_retract_node` responds to the `/arm/retract` service immediately with success without sending any joint commands.

**Motor dead zone compensation** — Real Kobuki motors ignore small commands. The DWA params set `vel_deadzone_linear: 0.1` and `vel_deadzone_angular: 0.5` so any non-zero velocity command is boosted above the threshold automatically.

**Topic names** — Real robot drivers omit the `turtlebot/` namespace prefix on some frames (`rplidar` not `turtlebot/rplidar`, `camera_link` not `turtlebot/camera_link`). The real robot config `exploration_real_params.yaml` sets `laser_frame: rplidar` and `map_frame: odom` to match.

### Useful Commands

```bash
# Teleoperate (manual override for testing)
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r __ns:=/turtlebot

# Shutdown Kobuki base safely
ros2 service call /turtlebot/turtlebot_shutdown std_srvs/srv/Trigger {}

# Retract arm manually
ros2 service call /arm/retract std_srvs/srv/Trigger {}

# Inspect TF tree
ros2 run tf2_tools view_frames
```

---

## Configuration

All tuneable parameters are in `src/online_motion_planning/config/exploration_params.yaml`.

### Frontier Detection (`frontier_node`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_frontier_dist_m` | 0.2 m | Discard frontiers closer than this to the robot |
| `visited_frontier_radius_m` | 0.5 m | Suppress previously visited frontiers within this radius |
| `use_global_search_window` | true | Fixed world bounding box; false = robot-centred growing square |
| `global_x_min/max` | −3 / 3 m | World-frame X bounds of the exploration area |
| `global_y_min/max` | −5 / 1 m | World-frame Y bounds of the exploration area |

### Path Planner / BiRRT\* (`path_planner_tb`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `acceptance_radius` | 0.1 m | Waypoint considered reached within this radius |
| `max_iterations` | 4000 | BiRRT\* iterations before declaring a goal unreachable |
| `max_consecutive_rejections` | 6 | Failed goals before stopping exploration |
| `goal_timeout_sec` | 30 s | Timeout waiting for a new frontier goal |
| `stuck_timeout_sec` | 9999 | Seconds without progress before rejecting the current waypoint |
| `stuck_regression_threshold` | 9999 | Distance regression (m) triggering immediate waypoint rejection |

### DWA (`dwa_service_node`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `use_odom_velocity` | false (sim) | Use actual odom velocity for dynamic window; set true on real robot |
| `adaptive_horizon` | true | Shorten prediction horizon when close to the goal |
| `use_depth_gate` | false (sim) | Block DWA output if no fresh depth frame has arrived |
| `vel_deadzone_linear/angular` | 0.0 | Minimum command needed to move the robot (non-zero on real hardware) |

### Global Costmap (`global_costmap`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `inflation_radius` | 0.2 m | Obstacle expansion radius |
| `map_frame` | `world_enu` | TF frame for the map origin |

---

## Topic Reference

| Topic | Type | Producer | Consumer |
|-------|------|----------|---------|
| `/turtlebot/camera/color/image_color` | `sensor_msgs/Image` | Stonefish | `image_crop_node` |
| `/turtlebot/camera/depth/image_depth` | `sensor_msgs/Image` | Stonefish | `image_crop_node` |
| `/turtlebot/camera/depth/image_cropped` | `sensor_msgs/Image` | `image_crop_node` | RTAB-Map |
| `/turtlebot/camera/depth/image_scan_cropped` | `sensor_msgs/Image` | `image_crop_node` | `depthimage_to_laserscan` |
| `/turtlebot/fake_scan` | `sensor_msgs/LaserScan` | `depthimage_to_laserscan` | RTAB-Map, DWA |
| `/turtlebot/sensors/imu_data` | `sensor_msgs/Imu` | Stonefish (NED) | `imu_ned_to_enu` |
| `/turtlebot/sensors/imu_enu` | `sensor_msgs/Imu` | `imu_ned_to_enu` | EKF node |
| `/turtlebot/odom` | `nav_msgs/Odometry` | Stonefish | EKF node |
| `/odometry/filtered` | `nav_msgs/Odometry` | EKF node | RTAB-Map, path planner |
| `/map` | `nav_msgs/OccupancyGrid` | RTAB-Map | `grid_mapping`, frontier node |
| `/inflated_map` | `nav_msgs/OccupancyGrid` | `grid_mapping` | Frontier node, path planner |
| `/inflated_map_dwa` | `nav_msgs/OccupancyGrid` | `dwa_local_costmap` | DWA service |
| `/frontier_goal` | `geometry_msgs/PoseStamped` | `frontier_node` | `path_planner_tb` |
| `/frontier/trigger` | `std_msgs/Bool` | `path_planner_tb` | `frontier_node` |
| `/frontier/goal_reached` | `geometry_msgs/PoseStamped` | `path_planner_tb` | `frontier_node` |
| `/frontier/exploration_complete` | `std_msgs/Bool` | `frontier_node` | `path_planner_tb` |
| `/turtlebot/cmd_vel` | `geometry_msgs/Twist` | `path_planner_tb` | Stonefish / Kobuki |
| `/arm/is_retracted` | `std_msgs/Bool` | `arm_retract_node` | `path_planner_tb` |

### DWA Service Interface

```
/dwa/compute_velocity  (dwa_interfaces/srv/ComputeVelocity)

Request:   float64 goal_x
           float64 goal_y
Response:  float64 linear_x
           float64 angular_z
           bool    success
```

---

## TF Frame Tree

```
world_enu (map frame)
  └── turtlebot/base_footprint      ← published by RTAB-Map / EKF
        ├── turtlebot/base_link
        ├── turtlebot/rplidar        ← laser scan frame
        ├── turtlebot/camera_link    ← RealSense RGB-D origin
        └── turtlebot/imu_enu        ← converted IMU frame
```

Stonefish also publishes a `world_ned` frame (NED convention). All planning and SLAM runs in `world_enu`.

---

## Simulation vs Real Robot

| Aspect | Simulation | Real Robot |
|--------|-----------|-----------|
| Map frame | `world_enu` | `odom` |
| Laser frame | `turtlebot/rplidar` | `rplidar` |
| Map topic | `/map` (RTAB-Map direct) | `/map` |
| DWA `use_odom_velocity` | `false` (sim odom reports 0) | `true` |
| DWA `use_depth_gate` | `false` | `true` |
| Arm retraction | Active (`is_sim: true`) | No-op (no arm fitted) |
| IMU conversion | `imu_ned_to_enu` node needed | Raw IMU already ENU |
| Localization | `robot_localization` EKF | `robot_localization` EKF (or custom `localization` node) |

---

## Package Structure

```
src/
├── online_motion_planning/         ← Main package
│   ├── launch/
│   │   ├── exploration_sim.launch.py        ← Simulation entry point
│   │   └── turtlebot_real_test.launch.py    ← Real robot entry point
│   ├── config/
│   │   ├── exploration_params.yaml          ← All tunable parameters
│   │   └── ekf_sim_params.yaml              ← robot_localization EKF config
│   └── online_motion_planning/
│       ├── frontier_node.py                 ← Frontier detection + BFS + scoring
│       ├── path_planner_tb.py               ← BiRRT* + DWA orchestration
│       ├── bidirectional_rrt_star.py        ← BiRRT* algorithm
│       ├── rrt_star.py                      ← RRT* base class
│       ├── rrt.py                           ← RRT base class
│       ├── imu_ned_to_enu.py                ← NED→ENU IMU frame converter
│       └── arm_retract_node.py              ← Retracts SwiftPro arm on startup
│
├── dwa_planner/                    ← DWA local planner (ROS 2 service)
│   └── dwa_planner/
│       ├── control_tb.py           ← DWAServiceNode
│       └── occupancy_grid_local.py ← Local inflated costmap for DWA
│
├── dwa_interfaces/                 ← ComputeVelocity.srv
├── grid_mapping/                   ← /map → /inflated_map obstacle inflation
├── image_utils/                    ← Camera crop + RTAB-Map time-sync
├── localization/                   ← Custom EKF odometry (real robot)
├── scan_to_cloud2/                 ← LaserScan → PointCloud2
├── depthimage_to_laserscan/        ← Depth image → fake LaserScan (vendored)
├── rtabmap/ + rtabmap_ros/         ← Visual SLAM (vendored)
├── turtlebot_simulation/           ← Stonefish scenarios + launch files
└── *_description/                  ← URDF robot models
```
