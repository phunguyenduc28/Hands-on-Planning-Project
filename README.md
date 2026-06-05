# Hands-on Planning Project

Autonomous exploration of an unknown environment using a **Kobuki TurtleBot 2** with an **Intel RealSense D435i** camera, running in [Stonefish](https://stonefish.readthedocs.io) simulation and deployable on the real robot.

The robot selects unexplored frontiers, plans collision-free paths with **Bidirectional RRT\***, and executes them using a **Dynamic Window Approach (DWA)** local planner. In simulation, **RTAB-Map** handles visual SLAM and occupancy mapping. On the real robot, a lightweight **scan-based occupancy map** (`scan_map_global`) is used instead, built directly from a fake 2D laser scan derived from the depth camera.

**Team:** Haadi / Huy / Phu

---

## System Architecture

### Simulation

```
┌──────────────────────────── Stonefish Simulator ──────────────────────────┐
│   RGB image    Depth image    Odometry    IMU (NED)                       │
└──────┬──────────────┬──────────────┬────────────┬──────────────────────── ┘
       │              │              │            │
  ImageCropNode  ImageCropNode   EKF Node    ImuNedToEnu
  (color crop)  (scan crop)  (robot_local.)  (NED→ENU)
       │              │              ▲            │
       │     depthimage_to_          │            │
       │      laserscan           /odom    /imu_enu
       │   → /turtlebot/fake_scan   └────────────┘
       │              │            /odometry/filtered
       └──────────────┴──────────────────┬────────────
                                         ▼
                                   RTAB-Map (Visual SLAM)
                               RGB-D + fake scan + EKF odom
                                    → 3D point cloud + /map
                                         │
                                       /map
                                         │
                               GlobalCostmap (grid_mapping)
                               occupancy_grid_original.py
                               (inflate /map by 0.2 m)
                                         │
                                   /inflated_map
                                ┌────────┴────────┐
                           FrontierNode     PathPlannerNode
                          (BFS + scoring)   (BiRRT* global)
                                │                  │
                         /frontier_goal ──► DWA Service
                                          (local velocity)
                                                   │
                                        /turtlebot/cmd_vel
```

### Real Robot

RTAB-Map was not used for planning on the real robot due to timestamp synchronisation issues between the RGB, depth, and scan streams. Instead, `scan_map_global` builds both the raw occupancy map and the inflated costmap in a single node, directly from the fake 2D laser scan produced by `depthimage_to_laserscan`. There is no IMU frame conversion — the real robot's IMU already publishes in ENU.

```
RealSense D435i
   └── Depth image
          │
     ImageCropNode
     (scan crop only — top + bottom fraction removed)
          │
    /camera/depth/image_scan_cropped
          │
   depthimage_to_laserscan
          │
   /turtlebot/fake_scan (2D LaserScan)
          │
   scan_map_global (grid_mapping)         Kobuki base
   log-odds ray-casting occupancy map   → /turtlebot/odom
   publishes /map_scan + /inflated_map_scan  │
          │                              EKF Node
          │                         (robot_localization)
          │                          /odometry/filtered
          │
   ┌──────┴───────┐
FrontierNode  PathPlannerNode
(BFS + scoring)  (BiRRT* global)
      │                │
/frontier_goal ──► DWA Service
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
| `grid_mapping` | **Sim:** `occupancy_grid_original` subscribes to RTAB-Map's `/map`, inflates it, publishes `/inflated_map`. **Real:** `scan_map_global` builds a full log-odds occupancy map directly from the fake laser scan and publishes both `/map_scan` and `/inflated_map_scan` in one node |
| `localization` | Custom differential-drive EKF (wheel encoders + IMU) — used on the real robot |
| `frontier_based_exploration` | Standalone simpler frontier node (legacy; the main pipeline uses `frontier_node` in `online_motion_planning`) |

### Sensor / Utility Packages

| Package | Purpose |
|---------|---------|
| `image_utils` | Crops RGB and depth images. **Sim:** produces both a color crop (→ RTAB-Map) and a depth scan crop (→ `depthimage_to_laserscan`). **Real:** only the depth scan crop is used in the active planning pipeline |
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

`image_crop_node` takes the full-resolution RGB and depth from the RealSense D435i and produces two cropped variants from the depth stream:

- **`image_cropped`** — bottom fraction removed (cuts the floor) → used by RTAB-Map **in simulation only**
- **`image_scan_cropped`** — top and bottom removed (cuts the robot arm and floor) → fed to `depthimage_to_laserscan`

`depthimage_to_laserscan` converts that narrow depth slice into a `LaserScan` on `/turtlebot/fake_scan`. This fake scan is the **only sensor input used for occupancy mapping and planning** on both simulation and real robot.

**IMU (simulation only):** Stonefish publishes the IMU in NED (North-East-Down) convention. `imu_ned_to_enu` negates yaw and ω_z before the data reaches the EKF. The real robot's IMU already publishes in ENU — no conversion node is needed.

The `robot_localization` EKF node fuses wheel odometry (`/turtlebot/odom`) with the (converted) IMU heading to produce `/odometry/filtered`, which is used by RTAB-Map and DWA.

### 2. Mapping

The mapping approach differs significantly between simulation and real robot:

**Simulation — RTAB-Map:**

RTAB-Map receives the cropped RGB-D images, fake laser scan, and EKF-fused odometry. It runs visual-inertial SLAM and publishes:
- `/map` (`OccupancyGrid`) — live 2D occupancy map used for planning
- A 3D point cloud of the environment
- TF: `world_enu → turtlebot/base_footprint`

The `occupancy_grid_original` node (in `grid_mapping`) subscribes to RTAB-Map's `/map` and publishes `/inflated_map` with all obstacles expanded by `inflation_radius` (0.2 m). This is what frontier detection and BiRRT\* consume.

**Real Robot — Scan-based map (`scan_map_global`):**

RTAB-Map was not reliable for planning on the real robot due to timestamp synchronisation issues between the RGB, depth, and scan streams. Instead, `scan_map_global` (in `grid_mapping`) builds the occupancy map **entirely from the fake 2D laser scan**, using log-odds ray-casting with Bresenham line tracing. It publishes both the raw map and the inflated costmap in a single node:
- `/map_scan` — raw log-odds occupancy grid
- `/inflated_map_scan` — obstacle-inflated version used directly by frontier detection and BiRRT\*

Frontier node and path planner are remapped to consume `/map_scan` / `/inflated_map_scan` instead of `/map` / `/inflated_map`.

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

#### Notes

- RTAB-Map is still launched (for 3D point cloud visualisation) but its `/map` output is **not** used for planning. All planning runs off `/map_scan` and `/inflated_map_scan` from `scan_map_global`.
- No `imu_ned_to_enu` node — the real IMU is already ENU.
- No arm retract node — `is_sim: false` causes the service to return success immediately without sending any joint commands.
- Motor dead zone: `vel_deadzone_linear: 0.1` and `vel_deadzone_angular: 0.5` boost small DWA commands above the Kobuki's dead band.
- TF frame names use no `turtlebot/` prefix on real hardware (`rplidar`, `camera_link`, `odom`).

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
| Mapping node | RTAB-Map → `occupancy_grid_original` | `scan_map_global` (depth scan only — RTAB-Map sync issues) |
| Raw map topic | `/map` | `/map_scan` |
| Inflated map topic | `/inflated_map` | `/inflated_map_scan` |
| Image crop used for planning | `image_scan_cropped` only | `image_scan_cropped` only |
| Image crop used for SLAM | `image_cropped` → RTAB-Map | not used for planning |
| IMU conversion | `imu_ned_to_enu` (NED→ENU) | not needed (real IMU is ENU) |
| Arm retraction | Active (`is_sim: true`) | No-op (no arm fitted) |
| DWA `use_odom_velocity` | `false` (sim odom reports 0) | `false` |
| DWA `use_depth_gate` | `false` | `true` |
| Motor dead zone | 0 | `linear: 0.1 m/s`, `angular: 0.5 rad/s` |
| Localization | `robot_localization` EKF (odom + IMU) | `robot_localization` EKF (odom + IMU) |

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
├── grid_mapping/                   ← Sim: inflate RTAB-Map /map; Real: scan_map_global (builds + inflates from fake scan)
├── image_utils/                    ← Camera crop (depth scan crop used in both; color crop used in sim only)
├── localization/                   ← Custom EKF odometry (real robot)
├── scan_to_cloud2/                 ← LaserScan → PointCloud2
├── depthimage_to_laserscan/        ← Depth image → fake LaserScan (vendored)
├── rtabmap/ + rtabmap_ros/         ← Visual SLAM (vendored)
├── turtlebot_simulation/           ← Stonefish scenarios + launch files
└── *_description/                  ← URDF robot models
```
