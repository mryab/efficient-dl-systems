# Containerized server

* Put `vgg16.pt` inside this directory

* Install Docker and Docker-compose

  * On Ubuntu
    ```bash
    curl https://get.docker.com -L | bash
    sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    ```

  * On other systems
    * [Docker](https://docs.docker.com/engine/install)
    * [Docker-compose](https://docs.docker.com/compose/install/)

* Build and run a simple server:

    If you are using MacOS
    ```bash
    export DOCKER_BUILDKIT=0
    export COMPOSE_DOCKER_CLI_BUILD=0
    ```

* Build and run a simple Python server:
    
    ```bash
    docker-compose up --build
    ```

* Build and run a production-ready server:

    ```bash
    docker-compose -f docker-compose.production.yaml up --build
    ```

## Further reading

* Docker - https://docs.docker.com/
* Gunicorn - https://docs.gunicorn.org/en/stable/index.html
* Nginx - https://nginx.org/ru/ 
* Why gunicorn & nginx - https://docs.gunicorn.org/en/stable/deploy.html
* Supervisord - http://supervisord.org/
* GIL - https://habr.com/ru/post/84629/

