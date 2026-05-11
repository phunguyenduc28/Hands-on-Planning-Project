#!/usr/bin/env python3
"""
System Validation Script
Checks if all components are properly installed and can be imported.
"""
import sys
import subprocess
import os


def check_command_exists(cmd):
    """Check if a command is available in PATH."""
    result = subprocess.run(['which', cmd], capture_output=True)
    return result.returncode == 0


def check_python_module(module_name, package_name=None):
    """Check if a Python module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_ros_package(package_name):
    """Check if a ROS 2 package can be found."""
    result = subprocess.run(
        ['ros2', 'pkg', 'prefix', package_name],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout.strip() if result.returncode == 0 else result.stderr


def main():
    print("=" * 70)
    print("NAVIGATION SYSTEM VALIDATION SCRIPT")
    print("=" * 70)
    
    checks_passed = 0
    checks_failed = 0
    
    # Check 1: ROS 2 Installation
    print("\n[1/8] Checking ROS 2 Installation...")
    if check_command_exists('ros2'):
        print("  ✓ ROS 2 command found")
        checks_passed += 1
    else:
        print("  ✗ ROS 2 not found in PATH")
        checks_failed += 1
    
    # Check 2: Python modules
    print("\n[2/8] Checking Python Dependencies...")
    required_modules = [
        ('rclpy', 'rclpy'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('yaml', 'pyyaml'),
        ('PIL', 'pillow'),
    ]
    
    for module, package in required_modules:
        success, error = check_python_module(module, package)
        if success:
            print(f"  ✓ {package:15} found")
            checks_passed += 1
        else:
            print(f"  ✗ {package:15} NOT FOUND - pip install {package}")
            checks_failed += 1
    
    # Check 3: ROS packages
    print("\n[3/8] Checking ROS 2 Message Packages...")
    ros_packages = [
        'geometry_msgs',
        'nav_msgs',
        'sensor_msgs',
        'visualization_msgs',
        'std_srvs',
    ]
    
    for pkg in ros_packages:
        success, path = check_ros_package(pkg)
        if success:
            print(f"  ✓ {pkg}")
            checks_passed += 1
        else:
            print(f"  ✗ {pkg} NOT FOUND")
            checks_failed += 1
    
    # Check 4: lab1_1 package
    print("\n[4/8] Checking lab1_1 Package...")
    success, path = check_ros_package('lab1_1')
    if success:
        print(f"  ✓ lab1_1 package found at: {path}")
        checks_passed += 1
        lab1_1_path = path
    else:
        print(f"  ✗ lab1_1 package not found - run: colcon build --packages-select lab1_1")
        checks_failed += 1
        lab1_1_path = None
    
    # Check 5: Script files exist
    print("\n[5/8] Checking Script Files...")
    if lab1_1_path:
        scripts = [
            'map_saver.py',
            'path_planner_rrt_star.py',
            'node_waypoint_controller.py',
        ]
        
        for script in scripts:
            script_path = os.path.join(lab1_1_path, 'lib', 'lab1_1', script)
            alt_path = os.path.join(lab1_1_path, 'lab1_1', script)
            if os.path.exists(script_path) or os.path.exists(alt_path):
                print(f"  ✓ {script}")
                checks_passed += 1
            else:
                print(f"  ✗ {script} not found")
                checks_failed += 1
    
    # Check 6: Maps directory
    print("\n[6/8] Checking Maps Directory...")
    maps_dir = os.path.expanduser('~/maps')
    if os.path.exists(maps_dir):
        print(f"  ✓ Maps directory exists: {maps_dir}")
        saved_maps = [f for f in os.listdir(maps_dir) if f.endswith('.yaml')]
        if saved_maps:
            print(f"    Found {len(saved_maps)} saved map(s)")
        checks_passed += 1
    else:
        print(f"  ! Maps directory doesn't exist yet (will be created on first save): {maps_dir}")
        checks_passed += 1  # Not critical
    
    # Check 7: Documentation
    print("\n[7/8] Checking Documentation...")
    docs = [
        'NAVIGATION_GUIDE.py',
        'QUICK_REFERENCE.md',
        'ARCHITECTURE.md',
        'IMPLEMENTATION_SUMMARY.md',
    ]
    
    if lab1_1_path:
        for doc in docs:
            doc_path = os.path.join(lab1_1_path, doc)
            if os.path.exists(doc_path):
                print(f"  ✓ {doc}")
                checks_passed += 1
            else:
                print(f"  ! {doc} not found")
    
    # Check 8: Summary
    print("\n[8/8] System Status Summary:")
    print("=" * 70)
    total = checks_passed + checks_failed
    if checks_failed == 0:
        print(f"✓ ALL CHECKS PASSED ({checks_passed}/{total})")
        print("\nYour system is ready to use!")
        return 0
    else:
        print(f"⚠ {checks_failed} check(s) failed ({checks_passed}/{total} passed)")
        print("\nSome components are missing. Run:")
        print("  pip install scipy pyyaml pillow")
        print("  cd ~/ROS2_Crash_Course/ros2_ws")
        print("  colcon build --packages-select lab1_1")
        print("  source install/setup.bash")
        return 1


if __name__ == '__main__':
    sys.exit(main())
