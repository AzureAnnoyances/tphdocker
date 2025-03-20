Build Docker image

```bash
docker build \
--ssh github_ssh_key=/home/wawj-u/.ssh/id_ed25519 \
-t dschng/tph -f Dockerfile .

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
-v /home/wawj-u/Documents/datasets/pcd:/root/pcds \
dschng/tph /bin/bash

```

Run the code
```bash
. /opt/installConda/CloudComPy310/bin/condaCloud.sh activate CloudComPy310 &&
cd /root/sdp_tph/main/ && git fetch && git switch testings_a
git pull --recurse-submodules && python3 main2.py /root/pcds/ p01e_B .las

```

Rerun code and clear data file
```bash
cd /root/pcds/p01e_B && rm -r ransac_data && cd /root/sdp_tph/main &&
git pull --recurse-submodules && python3 main2.py /root/pcds/ p01e_B .las

```