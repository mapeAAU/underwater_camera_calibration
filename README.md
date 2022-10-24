## Underwater Camera Calibration and 3D Reconstruction Based on Ray Tracing using Snell's Law
---

This is the repo for the source code and examples for the 3D reconstruction approach described in the paper *Camera Calibration for Underwater 3D Reconstruction Based on Ray Tracing using Snell's Law*. 
Below you can find an installation guide that ensures that you have the right packages and libraries installed for Python in order to run the code and examples.

Windows 7/10 Installation Guide
---
The first step is to install Python. 
This can be done in many ways and some prefer one approach over another. 
In this installation guide, we will take basis in Conda, which is a cross-platform package-, dependency- and environment-manager for Python. 
You can read more about it [here](https://conda.io/docs/index.html). 
Python will be installed automatically when you install Conda.

1. Download Anaconda [here](https://repo.anaconda.com/archive/Anaconda3-5.1.0-Windows-x86_64.exe).
2. Run the .exe file and follow the instructions on screen.
3. Open the Anaconda Prompt (e.g., from the start menu).
4. Update Anaconda by running ```$conda update conda```

The next step is to create a Python environment with the packages and dependencies needed for running the code. 
If you are new to Python and virtual environments, you can read more about what it is and why it can be a good idea to use it [here](https://docs.python.org/3/tutorial/venv.html). 
We will create an environment named **ucc** (for Underwater Camera Calibration). 
In the Anaconda prompt write the following:

1. ```$conda create --name ucc```
2. ```$activate ucc```

Now that the environment has been activated, we can install the packages that we need in order to run the camera calibration:

1. ```$conda install numpy anaconda-client pandas matplotlib scikit-learn```
2. ```$conda install -c anaconda opencv```

And that's it! If everything went smooth you should be able to run the calibration examples. **Note:** remember to activate the ucc-environment every time you want to run the code as the packages are only installed under this environment.

Ubuntu (16.04 and 18.04) Installation Guide
---
Python is installed on Ubuntu 16.04 as default and if you are familiar to Python, the only thing you need to do in order to run the code-examples is to get OpenCV3 and the rest of the dependencies. However, if you are new to Python you can follow this guide and get up and running in no time.

The first thing we will do is to download and install Anaconda, which is a cross-platform package-, dependency- and environment-manager for Python - you can read more about it [here](https://conda.io/docs/index.html).

1. Download Anaconda for Python 3.6 [here](https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh)
2. Move the downloaded file to your home folder
3. Open a terminal in your home folder and write ```bash ./Anaconda3-5.1.0-Linux-x86_64.sh``` (or whatever version of Anaconda you have downloaded)
4. Follow the instructions on screen
    * Say 'yes' when it prompts for prepending the Anaconda install location to PATH
    * You do not need to install the Microsoft VS Code
5. Close the terminal

Next step is to update Conda and create a virtual environment for your underwater camera calibration code.
If you are new to Python and virtual environments, you can read more about what it is and why it can be a good idea to use it [here](https://docs.python.org/3/tutorial/venv.html).
We will create an environment named **ucc** (for Underwater Camera Calibration).
Open a new terminal and write the following:

1. ```$conda update conda```
2. ```$conda create --name ucc```
3. ```$source activate ucc``` 

Now you should have an active Python 3.6 environment and the only thing that is missing is the respective libraries needed to run the code.
We install them by running the following lines:

1. ```$conda install numpy anaconda-client pandas matplotlib scikit-learn```
2. ```$conda install -c anaconda opencv```

And that's it! If everything went smooth you should be able to run the calibration examples. 
