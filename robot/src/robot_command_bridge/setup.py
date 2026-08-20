from setuptools import find_packages, setup

package_name = 'robot_command_bridge'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hs',
    maintainer_email='cyok7425@naver.com',
    description='AWS IoT MQTT MOVE_TO command bridge for Nav2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            (
                'robot_command_bridge_node = '
                'robot_command_bridge.'
                'robot_command_bridge_node:main'
            ),
        ],
    },
)
