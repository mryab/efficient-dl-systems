You can use any cloud provider to get some machines to add to the swarm. 
The following example shows how to use docker swarm with `docker-machine` locally.

Install Docker machine from here: https://github.com/docker/machine

See https://github.com/Nordstrom/docker-machine/blob/master/docs/install-machine.md

```bash
docker-machine create --driver virtualbox node1
docker-machine create --driver virtualbox node2
docker-machine create --driver virtualbox node3

docker-machine ls

eval `docker-machine env node1`

docker swarm init --advertise-addr 192.168.99.110

eval `docker-machine env node2`

docker swarm join --token SWMTKN-1-TOKEN 192.168.99.107:2377

eval `docker-machine env node3`

docker swarm join --token SWMTKN-1-TOKEN 192.168.99.107:2377
```

Create a service

```bash
eval `docker-machine env node1`

docker service create \
  --name=viz \
  --publish=8080:8080/tcp \
  --constraint=node.role==manager \
  --mount=type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  dockersamples/visualizer

docker ps
docker service ls
```

Open `192.168.99.107:8080`

```bash
eval `docker-machine env node1`

docker stack deploy --compose-file docker-compose.swarm.yaml fileservice
```

Open `192.168.99.107:9090`