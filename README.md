Build Docker image
## New Version
```bash
export docker_name="tphv2"
docker build \
--ssh github_ssh_key=/home/ds1804/.ssh/id_ed25519 \
-t $docker_name -f Dockerfile .
```
```bash
xhost local:docker

docker run -it \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /usr/bin/docker:/usr/bin/docker \
--net=host \
--gpus all \
--privileged \
--volume /dev:/dev \
--volume /tmp/.x11-unix:/tmp/.x11-unix \
--volume ~/.ssh/ssh_auth_sock:/ssh-agent \
--env SSH_AUTH_SOCK=/ssh-agent \
--env DISPLAY=$DISPLAY \
--env TERM=xterm-256color \
-v /home/ds1804/pcds:/root/pcds \
$docker_name /bin/bash
```
```bash
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/10ounVnH2i16FWl3WK4alm0YOAGsuH__f?usp=sharing
cd /root/sdp_tph/ && git fetch && git switch testings_a && git pull --recurse-submodules && 
cd /root/sdp_tph/main/ && python3 main2.py /root/pcds/ Tangkak_1 .laz
```

## Old Version
```bash
cd /root/sdp_tph/main
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/10ounVnH2i16FWl3WK4alm0YOAGsuH__f?usp=sharing
. /opt/installConda/CloudComPy310/bin/condaCloud.sh activate CloudComPy310 &&
cd /root/sdp_tph/ && git fetch && git switch testings_a && git pull --recurse-submodules && 
cd /root/sdp_tph/main/ && python3 main2.py /root/pcds/ Tangkak_1 .laz
```
Run Docker container and open the Docker container terminal 
```bash
xhost local:docker

cd && docker run -it \
-v /var/run/docker.sock:/var/run/docker.sock \
-v /usr/bin/docker:/usr/bin/docker \
--net=host \
--gpus all \
--privileged \
--volume /dev:/dev \
--volume /tmp/.x11-unix:/tmp/.x11-unix \
--volume ~/.ssh/ssh_auth_sock:/ssh-agent \
--env SSH_AUTH_SOCK=/ssh-agent \
--env DISPLAY=$DISPLAY \
--env TERM=xterm-256color \
-v /home/ds1804/pcds:/root/pcds \
dschng/tph /bin/bash

```

Run the code
```bash
. /opt/installConda/CloudComPy310/bin/condaCloud.sh activate CloudComPy310 &&
cd /root/sdp_tph/ && git fetch && git switch testings_a && git pull --recurse-submodules && 
cd /root/sdp_tph/main/ && python3 main2.py /root/pcds/ Tangkak_1 .laz

```

Single Tree
```bash
. /opt/installConda/CloudComPy310/bin/condaCloud.sh activate CloudComPy310 &&
cd /root/sdp_tph/ && git fetch && git switch testings_a && git pull --recurse-submodules && 
cd /root/sdp_tph/main/ && python3 main_single_tree.py /root/pcds/single_tree/ model_dense4k10fps1Tree .ply
```

```
. /opt/installConda/CloudComPy310/bin/condaCloud.sh activate CloudComPy310 &&
python3 
```


Rerun code and clear data file
```bash
sudo chown -R ds1804 p01e_B

cd /root/pcds/p01e_B && rm -r ransac_data && cd /root/sdp_tph/main &&
git pull --recurse-submodules && python3 main2.py /root/pcds/ p01e_B .las

```