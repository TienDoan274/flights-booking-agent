link crawl chuyến bay: "https://vietnamairport.vn/thong-tin-lich-bay"

link iata airlines code: "https://github.com/benct/iata-utils"

link iata airports code: "https://github.com/lxndrblz/Airports"

### Download spark on local:

https://spark.apache.org/downloads.html

### Set environment variable:

SPARK_HOME: C:\Program Files\Spark\spark-3.5.3-bin-hadoop3 (for example)

docker compose up

### Crawling and ingest to mongodb.
Tạo topic flights trong kafka.
docker exec -it kafka sh

../../bin/kafka-topics --create --topic flights \
                --bootstrap-server localhost:9092 \
                --partitions 1 \
                --replication-factor 1

python ./ingest_data_consumer/consumer-mongo.py

python ./crawlers_producer/crawl_flights.py

### Running chat bot.

python ./agent/Chatbot.py





