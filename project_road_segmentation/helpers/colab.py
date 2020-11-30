import subprocess


def mount_and_pull(branch_name, drive, os):
    mgc = get_ipython().magic
    execute_and_get_output('git fetch')
    execute_and_get_output('git reset --hard')
    execute_and_get_output('git checkout {}'.format(branch_name))
    execute_and_get_output('git pull')
    execute_and_get_output('git branch')
    mgc('cd project_road_segmentation')
    execute_and_get_output('ls')


def execute_and_get_output(command):
    result = subprocess.check_output(command, shell=True)
    print(result.decode('utf-8'))


def download_model(model_name, save_dir, files):
    zip_cmd = 'zip -r ./{}.zip ./{}'.format(model_name, save_dir)
    execute_and_get_output(zip_cmd)
    files.download('./{}.zip'.format(model_name))

# Add these lines to the notebook to pull your branch on the Drive
# if COLAB:
#     from helpers.colab import mount_and_pull
#     from google.colab import drive
#     drive.mount('/content/drive')
#     drive_path = '/content/drive/Shareddrives/ML_Road_Segmentation/CS-433-project-2'
#     os.chdir(drive_path)
#     BRANCH_NAME = 'nom_de_votre_branche'
#     mount_and_pull(BRANCH_NAME, drive, os)
#
# And these lines after saving your model to download it on your local system
# (MODEL_NAME and the save directory should be set in the Notebook)
# if COLAB:
#     from helpers.colab import download_model
#     from google.colab import files
#     download_model(MODEL_NAME, model_chosen['save_dir'], files)

