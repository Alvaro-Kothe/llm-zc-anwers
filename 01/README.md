# Solution

## Question 1

1. Start elasticsearch container:

   ```
   docker run --rm -it --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:8.17.6
   ```

1. Query it

   ```
   ❯ curl --silent localhost:9200 | jq ".version.build_hash"
   "dbcbbbd0bc4924cfeb28929dc05d82d662c527b7"
   ```

**Answer:** dbcbbbd0bc4924cfeb28929dc05d82d662c527b7

## Question 2

**Answer:** index

## Question 3

Running `python main.py` gives

```
[{'_index': 'course-questions', '_id': '3NqpZJcBS_hA7BL0Jhd4', '_score': 43.74824, '_source': {'text': 'Launch the contai...}]
```

**Answer:** 44.50

## Question 4

```
result = elastic_search(query, es_client, course_filter="machine-learning-zoomcamp")[2]
```

```
/xgb_model.bin", "./"]\t\t\t\t\t\t\t\t\t\t\tGopakumar Gopinathan', 'section': '5. Deploying Machine Learning Models', 'question': 'How do I copy files from a different folder into docker container’s working directory?', 'course': 'machine-learning-zoomcamp'}}
```

**Answer:** How do I copy files from a different folder into docker container’s working directory?

## Question 5

```
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: How do copy a file to a Docker container?

CONTEXT:
Q: How do I debug a docker container?
A: Launch the container image in interactive mode and overriding the entrypoint, so that it starts a bash command.
docker run -it --entrypoint bash <image>
If the container is already running, execute a command in the specific container:
docker ps (find the container-id)
docker exec -it <container-id> bash
(Marcos MJD)

Q: How do I copy files from my local machine to docker container?
A: You can copy files from your local machine into a Docker container using the docker cp command. Here's how to do it:
To copy a file or directory from your local machine into a running Docker container, you can use the `docker cp command`. The basic syntax is as follows:
docker cp /path/to/local/file_or_directory container_id:/path/in/container
Hrithik Kumar Advani

Q: How do I copy files from a different folder into docker container’s working directory?
A: You can copy files from your local machine into a Docker container using the docker cp command. Here's how to do it:
In the Dockerfile, you can provide the folder containing the files that you want to copy over. The basic syntax is as follows:
COPY ["src/predict.py", "models/xgb_model.bin", "./"]                                                                                   Gopakumar Gopinathan
1446
```

**Answer:** 1446

## Question 6

**Answer:** 320
