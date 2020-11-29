def mount_and_pull(branch_name):
    mgc = get_ipython().magic

    # Set current directory and clone github repo
    drive_path = '/content/drive/Shareddrives/ML_Road_Segmentation/'
    os.chdir(drive_path)
    repo = "CS-433-project-2"

    if not os.path.isdir("CS-433-project-2"):
        uname = input("Github username:")
        password = getpass('Password:')
        mgc("!git clone 'https://{}:{}@github.com/Julien-Ben/{}.git'.format(uname, password, repo)")
    else:
        os.chdir(repo)
        mgc('!git fetch')
        mgc('!git checkout {}'.format(branch_name))
        mgc('!git reset --hard')
        mgc('!git pull')
        mgc('!git branch')
    mgc('%cd project_road_segmentation')
    mgc('!ls')


def download_model(model_name, save_dir):
    from google.colab import files
    zip_cmd = 'zip -r ./{}.zip ./{}'.format(model_name, save_dir)
    mgc(zip_cmd)
    files.download('./{}.zip'.format(model_name))

