cd ~

# Download cuda installer
wget -O cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64

# Add public key
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Update and install
sudo apt-get update
sudo apt-get install cuda

# CUDA path to .bashrc
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Download cuDNN 7.4 for CUDA 10 from https://developer.nvidia.com/rdp/cudnn-archive
tar -xzvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

# install mujoco
mkdir .mujoco && cd .mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
rm mujoco200_linux.zip
# copy mjkey.txt file in .mujoco directory

# setup virtualenv
mkdir movo && cd movo
sudo apt-get install python-pip python3-dev
sudo pip install virtualenv
virtualenv -p python3 movo-env
echo "alias movo_env='source /home/$USER/movo/movo-env/bin/activate'" >> ~/.bash_aliases
echo "alias movo_env='source /home/$USER/movo/movo-env/bin/activate'" | sudo tee -a /root/.bash_aliases
source ~/.bashrc
movo_env

# install dm_control hardware rendering dependencies
sudo apt-get install libglfw3 libglew2.0

# install dm_control
pip install git+git://github.com/deepmind/dm_control.git

# install requirements.txt
pip install -r requirements.txt

# install pytorch
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl torchvision

#install kinova-api
git clone git@github.com:rikenmehta03/kinova-api.git
cd kinova-api
make
sudo make install
pip install .

# Test jaco api with super-user
sudo su
movo_env
python test.py # make sure jaco usb is connected to the system

