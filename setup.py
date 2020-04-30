from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym'))
from version import VERSION

# Environment-specific dependencies.
extras = {
  'atari': ['atari_py~=0.2.0', 'Pillow', 'opencv-python'],
  'box2d': ['box2d-py~=2.3.5'],
  'classic_control': [],
  'mujoco': ['mujoco_py>=1.50, <2.0', 'imageio'],
  'robotics': ['mujoco_py>=1.50, <2.0', 'imageio'],
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='gymn',
      version=VERSION,
      description='modified gym environment based on OpenAI gym',
      author='ylb',
      author_email='paladinee15@qq.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('gymn')],
      zip_safe=False,
      install_requires=[
          'scipy', 'numpy>=1.10.4', 'pyglet>=1.4.0,<=1.5.0', 'cloudpickle>=1.2.0,<1.4.0',
          'enum34~=1.1.6;python_version<"3.4"',
      ],
      extras_require=extras,
      package_data={'gymn': [
        'envs/mujoco/assets/*.xml',
        'envs/classic_control/assets/*.png',
        'envs/robotics/assets/LICENSE.md',
        'envs/robotics/assets/fetch/*.xml',
        'envs/robotics/assets/hand/*.xml',
        'envs/robotics/assets/stls/fetch/*.stl',
        'envs/robotics/assets/stls/hand/*.stl',
        'envs/robotics/assets/textures/*.png']
      },
      tests_require=['pytest', 'mock'],
      python_requires='>=3.5',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
)
