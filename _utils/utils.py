import os 
import sys
import pathlib

def isexistfeather(dirname):
    ''' Check if there is a feather file in that dir path (or sub-dir path) '''
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.feather':
                return True
    return False

def createbackupfolder():
    ''' If there is a feather file in the path, create a backup folder with the current time '''
    import shutil    
    from datetime import datetime
    # current_dir = pathlib.Path.cwd()
    current_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
    parent_dir = current_dir.parent
    dir_path = pathlib.Path.joinpath(parent_dir, 'result', 'eicu')
    if not isexistfeather(dir_path):
        return False
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    new_dir_path = pathlib.Path.joinpath(parent_dir, 'result', '_backup', date, 'eicu')
    shutil.move(dir_path, new_dir_path)
    return True