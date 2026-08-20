from setuptools import find_packages, setup

package_name = 'my_robot_ai_control'

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
    maintainer='hs',
    maintainer_email='tjdrnr614@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_collector = my_robot_ai_control.data_collector_node:main',
            'ai_cleaner = my_robot_ai_control.ai_cleaner_node:main',
        ],
    },
)
