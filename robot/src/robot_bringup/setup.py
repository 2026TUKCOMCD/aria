import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('lib', package_name, 'certs'), glob('robot_bringup/certs/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hs',
    maintainer_email='hs@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'zone_segmentation_node = robot_bringup.zone_segmentation_node:main',
            'air_purify_scheduler_node = robot_bringup.air_purify_scheduler_node:main',
            'aria_mqtt_node = robot_bringup.aria_mqtt_node:main',
            'aria_controller_node = robot_bringup.aria_controller_node:main',
            'aria_cooking_ai_node = robot_bringup.aria_cooking_ai_node:main',
        ],
    },
)
