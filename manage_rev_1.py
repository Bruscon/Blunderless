import os
import shutil
from datetime import datetime

def manage_folders():
    # Define paths
    current_dir = os.getcwd()  # Current working directory
    rev_1_folder = os.path.join(current_dir, 'rev_1')
    archive_folder = os.path.join(current_dir, 'archive')

    # Check if 'rev_1' folder exists
    if os.path.exists(rev_1_folder):
        # Create the archive folder if it doesn't exist
        if not os.path.exists(archive_folder):
            os.makedirs(archive_folder)

        # Generate a timestamp for the folder name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archived_folder_name = f"rev_1_{timestamp}"

        # Move 'rev_1' folder to 'archive' with a timestamped name
        shutil.move(rev_1_folder, os.path.join(archive_folder, archived_folder_name))
        print(f"Moved {rev_1_folder} to {os.path.join(archive_folder, archived_folder_name)}")

    # Recreate 'rev_1' folder
    os.makedirs(rev_1_folder)
    print(f"Created a new folder: {rev_1_folder}")

    # Create subfolders inside 'rev_1'
    os.makedirs(os.path.join(rev_1_folder, 'keep_an_eye_on'))
    os.makedirs(os.path.join(rev_1_folder, 'questionnaire'))
    os.makedirs(os.path.join(rev_1_folder, 'prompts'))
    os.makedirs(os.path.join(rev_1_folder, 'responses'))
    os.makedirs(os.path.join(rev_1_folder, 'position_history'))
    print("Created subfolders")

    shutil.copy('prompts/questions.txt', 'rev_1/questionnaire')
    print("Copied over questionaire")

