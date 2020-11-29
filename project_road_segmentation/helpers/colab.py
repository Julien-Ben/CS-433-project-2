def mount_and_pull(branch_name, drive, os):
    drive.mount('/content/drive')
    drive_path = '/content/drive/Shareddrives/ML_Road_Segmentation/CS-433-project-2'
    os.chdir(drive_path)
    mgc = get_ipython().magic
    os.system('git fetch')
    os.system('git reset --hard')
    os.system('git checkout {}'.format(branch_name))
    os.system('git pull')
    os.system('git branch')
    mgc('cd project_road_segmentation')
    os.system('ls')


def download_model(model_name, save_dir, files):
    zip_cmd = 'zip -r ./{}.zip ./{}'.format(model_name, save_dir)
    os.system(zip_cmd)
    files.download('./{}.zip'.format(model_name))

