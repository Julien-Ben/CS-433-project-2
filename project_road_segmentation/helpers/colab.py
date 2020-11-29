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

# Add these lines to the notebook to pull your branch on the Drive
# if COLAB:
#     from helpers.colab import mount_and_pull
#     from google.colab import drive
#     BRANCH_NAME = 'nom_de_votre_branche'
#     mount_and_pull(BRANCH_NAME, drive, os)
#
# And these lines after saving your model to download it on your local system
# (MODEL_NAME and the save directory should be set in the Notebook)
# if COLAB:
#     from helpers.colab import download_model
#     from google.colab import files
#     download_model(MODEL_NAME, model_chosen['save_dir'], files)

