import os
import shutil

folder_name = "ISBI2016_ISIC_Part3_Test_Data"

for file in os.listdir(os.path.join(folder_name, folder_name)):
    if file.endswith(".jpg"):
        shutil.move(
            os.path.join(folder_name, folder_name, file),
            os.path.join(folder_name, file),
        )