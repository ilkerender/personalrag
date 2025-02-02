# personalrag
A RAG AI assistant that helps answer questions based on your personal documents stored on Google Drive or local files. Uses deepseek r1 that runs locally.

## Installation

Instructions on how to install and set up the project.

```bash
# Clone the repository
git clone https://github.com/ilkerender/personalrag.git

# Navigate to the project directory
cd personalrag

# Install dependencies
pip install -r requirements.txt

# Install ollama & run deepseek r1:1.5b
./setup_ollama.sh 
Pulling model: deepseek-r1:1.5b...
pulling manifest 
...                  
verifying sha256 digest 
writing manifest 
success 
Starting Ollama server...
Ollama setup complete. Running at http://localhost:11434

# Check whether you can interact with it : 
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:7b",
  "messages": [{ "role": "user", "content": "Tell me a funny joke." }],
  "stream": false
}'
{"model":"deepseek-r1:7b","created_at":"2025-02-02T12:52:29.980823322Z","message":{"role":"assistant","content":"\u003cthink\u003e\nAlright, the user asked for a funny joke. I should come up with something simple and light-hearted.\n\nMaybe play on words related to the letter 'e'. \"Even\" is a good choice because it's easy to pronounce and has a humorous twist.\n\nHow about, \"Why don’t skeletons fight each other? They have no guts!\"\n\nThat plays on the word \"guts,\" which skeleton doesn't have, so it's funny without being too complicated.\n\u003c/think\u003e\n\nSure! Here's a joke for you:\n\nWhy don’t skeletons fight each other?  \nThey have no guts!"},"done_reason":"stop","done":true,"total_duration":19706875936,"load_duration":6084315831,"prompt_eval_count":9,"prompt_eval_duration":393000000,"eval_count":120,"eval_duration":13228000000}[ilker@almahost personal]$ 



#Install Milvus for content store
./setup_milvus.sh 
Downloading Milvus installation script...
Starting Milvus Docker container...
..
..
Milvus setup complete. Running at localhost:19530"

>docker ps 
CONTAINER ID   IMAGE                                      COMMAND                  CREATED        STATUS                 PORTS                                                                                                                                 NAMES
1180f5d12260   minio/minio:RELEASE.2020-12-03T00-03-10Z   "/usr/bin/docker-ent…"   3 hours ago    Up 3 hours (healthy)   9000/tcp                                                                                                                              milvus-minio
6956fb611e8e   quay.io/coreos/etcd:v3.5.0                 "etcd -advertise-cli…"   3 hours ago    Up 3 hours             2379-2380/tcp                                                                                                                         milvus-etcd
1b1596b0f17b   milvusdb/milvus:v2.5.4                     "/tini -- milvus run…"   16 hours ago   Up 3 hours (healthy)   0.0.0.0:2379->2379/tcp, :::2379->2379/tcp, 0.0.0.0:9091->9091/tcp, :::9091->9091/tcp, 0.0.0.0:19530->19530/tcp, :::19530->19530/tcp   milvus-standalone


#Generate sample bill data as json lines file
>python3.9 random_bills.py 
Generated 1000 documents in documents.jsonl


#Upload the documents to vector store
python3.9 milvus_upload.py --creds ../secrets.yaml.example --collection mybills --input documents.jsonl --out uploaded.data
[nltk_data] Downloading package stopwords to /home/ilker/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Connected to Milvus at localhost:19530
Loaded 1000 documents from documents.jsonl
Created collection 'mybills'.
Index created on 'content_embedding' field.
Collection 'mybills' is now loaded and ready for queries.



Now that our LLM + Vector Store is ready to go , start the Assistant app, to interact with :

>python3.9 milvus_upload.py --creds ../secrets.yaml.example --collection mybills --input documents.jsonl --out uploaded.data
[nltk_data] Downloading package stopwords to /home/ilker/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Connected to Milvus at localhost:19530
Loaded 1000 documents from documents.jsonl
Created collection 'mybills'.
Index created on 'content_embedding' field.
Collection 'mybills' is now loaded and ready for queries.

```

![image](https://github.com/user-attachments/assets/65df9eb0-f804-4ebc-9ef8-5da54624465d)

