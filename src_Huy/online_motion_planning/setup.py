from setuptools import find_packages, setup

package_name = 'online_motion_planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='huy',
    maintainer_email='dohuy9379@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'control_tb = online_motion_planning.control_tb:main',
            'occupancy_grid_original = online_motion_planning.occupancy_grid_original:main',
        ],
    },
)
