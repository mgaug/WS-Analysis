docker build -t ws .
docker images
docker run -it ws --name ws 

docker ps
docker exec -it ws bash

docker run -it ws --name ws -v $(pwd)/Results:Results/
