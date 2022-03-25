from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['accelnet_challenge_sdu'],
    package_dir={'': 'src'},
    # scripts=['']
)

setup(**d)
