from setuptools import find_packages, setup

package_name = 'charging_pose_initializer'

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
    maintainer_email='hs@example.com',
    description='Initialize AMCL pose from charging station API',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            (
                'charging_pose_initializer_node = '
                'charging_pose_initializer.'
                'charging_pose_initializer_node:main'
            ),
        ],
    },
)
