# Docker Container for PIE

Build Docker container image containing the PIE code base:

```
docker image build -t pie:0.1.0.flowserv .
```

Push container image to DockerHub.

```
docker image tag pie:0.1.0 heikomueller/pie:0.1.0.flowserv
docker image push heikomueller/pie:0.1.0.flowserv
```


Run Commands inside Docker Container
------------------------------------

Use the following command to run PIE single-image analysis in the Docker container.

```
docker container run \
    --rm \
    -v /home/user/projects/PIE/PIE_test_data:/data \
    pie:0.1.0.flowserv \
    pie run single -t brightfield /data/IN/SL_170619_2_GR_small/t01xy0001.tif /data/OUT
```

Run growth rate analysis experiment:

```
docker container run \
    --rm \
    -v /home/user/projects/PIE/PIE_test_data:/data \
    pie:0.1.0.flowserv \
    pie run experiment -t 10 /data/IN /data/OUT
```
