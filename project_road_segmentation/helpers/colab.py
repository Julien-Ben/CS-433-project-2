def mount_and_pull(branch_name):
    from google.colab import drive
    drive.mount('/content/drive')

    # Set current directory and clone github repo
    from getpass import getpass

    drive_path = '/content/drive/MyDrive/Colab_data/'
    os.chdir(drive_path)
    repo = "CS-433-project-2"

    if not os.path.isdir("CS-433-project-2"):
        uname = input("Github username:")
        password = getpass('Password:')
        !git clone 'https://{}:{}@github.com/Julien-Ben/{}.git'.format(uname, password, repo)
    else:
        checkout_string = 'git checkout {}'.format(branch_name)
        os.chdir(repo)
        !git fetch
        os.system(checkout_string)
        !git reset --hard
        !git pull
        !git branch
    %cd project_road_segmentation
    !ls

def download_model(model_name, save_dir):
    from google.colab import files
    zip_cmd = 'zip -r ./{}.zip ./{}'.format(model_name, save_dir)
    os.system(zip_cmd)
    files.download('./{}.zip'.format(model_name))

