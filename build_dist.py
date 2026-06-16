"""
Build the distribution
"""
import shutil
import os


# Clean dist folder
shutil.rmtree('./build/', ignore_errors=True)
shutil.rmtree('./dist/', ignore_errors=True)
shutil.rmtree('./aule.egg-info/', ignore_errors=True)

# Build distribution
os.system("python build_doc.py")
os.system("python -m build")
os.system("python -m twine upload dist/* --verbose")
