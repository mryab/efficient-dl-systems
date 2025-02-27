You can use any cloud provider to get some machines to add to the swarm. 
The previous years had a guide on how to use docker swarm with machines created locally via VirtualBox.
It used to operate using `docker-machine`, but it is now deprecated. You can check previous years for details.

YZ: In this guide for previous years, local machines were created using VirtualBox. 
Since I am a weirdo who uses WSL2, setupping everything properly was a pain in the ass.
If someone wants to replicate the setup, I **definitely** recommend the cloud way.
Using pure linux (non-WSL) may be easier but still not recommended.
What we'd need here is just some VM with Docker installed where we can ssh to, replicated 3 times.
Ensure the Docker version is the same, this is VERY important!
Also check firewalls (use ChatGPT for `sudo ufw allow icmp`-like commands and troubleshooting)

On machine 1, run:
```bash
docker swarm init --advertise-addr MACHINE_1_IP
```

On machines 2 and 3, run
```bash
docker swarm join --token SWMTKN-1-TOKEN MACHINE_1_IP:2377
```

If docker swarm commands hang, don't forget to check if the machines can ping each other and something is listening on port 2377
```shell
nc -zv MACHINE_1_IP 2377
```

Check that nodes are indeed available (on manager node, aka node1)
```shell
docker node ls
```

You should see all nodes.

Then, on machine 1,

```bash
docker service create \
  --name=viz \
  --publish=8080:8080/tcp \
  --constraint=node.role==manager \
  --mount=type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  dockersamples/visualizer

docker ps
docker service ls
```

For local setup, you might require to pass additional arguments and setup ingress so it listens to local addresses.
`docker network inspect ingress | grep Subnet` should yield something. Again, going through all this is *NOT* recommended.

Open `MACHINE1_IP:8080` (and because of ingress, `MACHINE2_IP:8080` should give same result too)

Ssh into machine1 (you should move the compose file there first) and run

```bash
docker stack deploy --compose-file docker-compose.swarm.yaml fileservice
```

Open `MACHINE1_IP:9090`
