import subprocess
import os

subprocess.run([
    r'C:\Users\jonah\Downloads\platform-tools-latest-windows\platform-tools\adb.exe',
    'pull',
    'sdcard/Download',
    r'C:\Users\jonah\Downloads\XYZ_data'
], check=True)

#7 new ones, first and second last excluded