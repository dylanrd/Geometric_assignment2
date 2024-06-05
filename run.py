# Running this with `blender --python run.py` launches blender with your plugins already enabled!

# Blender will actually run this in another directory, so we need to make sure everything is available to import
import os
import sys

sys.path.append(os.path.dirname(__file__))

# Make sure we have the packages we need
# This is necessary because Blender comes with its own copy of Python
# running `pip install -r requirements.txt` should enable code-completion in your IDE,
# but Blender needs to install the requirements itself.
import pip

pip.main(['install', '-r', f'{os.path.dirname(__file__)}/requirements.txt'])

# Add your plugins to the Blender UI
import assignment3
assignment3.register()
