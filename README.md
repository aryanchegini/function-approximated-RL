# RLCoursework

Environment Setup Instructions

Step 1: Create a virtual environment
In your project folder, run:
python3.11 -m venv rl-cw-env

Step 2: Activate the environment
source rl-cw-env/bin/activate

Step 3: Upgrade pip
pip install --upgrade pip

Step 4: Install the required packages
Install Gymnasium with Atari support, AgileRL, PyTorch, and ROM installer:
pip install "gymnasium[atari]" ale-py autorom agilerl torch pygame numpy

Step 5: Install Atari ROMs
Run this command to download the Atari game ROMs:
AutoROM --accept-license