from setuptools import setup

package_name = 'my_robot_map_export'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hs',
    maintainer_email='hs@todo.todo',
    description='Export OccupancyGrid to PNG and JSON',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'occupancy_to_png_json = my_robot_map_export.occupancy_to_png_json:main',
            'upload_map = my_robot_map_export.upload_map:main',
        ],
    },
)
