from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pymongo
from pymongo import MongoClient
from pyspark.sql.functions import to_json, struct
from datetime import datetime

def create_spark_session():
    return (SparkSession.builder
            .appName("FlightDataProcessor")
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate())

def process_stream(df, epoch_id):
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("scheduled_time", StringType(), True),
        StructField("updated_time", StringType(), True),
        StructField("route", StringType(), True),
        StructField("flight_id", StringType(), True),
        StructField("counter", StringType(), True),
        StructField("gate", StringType(), True),
        StructField("status", StringType(), True)
    ])
    
    try:
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")
        
        valid_df = parsed_df.filter(
            col("flight_id").isNotNull() &
            col("date").isNotNull() &
            col("scheduled_time").isNotNull()
        )
        
        rejected_df = parsed_df.subtract(valid_df)
        if rejected_df.count() > 0:
            print("Rejected records due to null values in primary key columns:")
            rejected_df.show(truncate=False)
        
        processed_df = valid_df \
            .withColumn("processing_timestamp", current_timestamp()) \
            .withColumn("departure_airport", 
                       when(col("Route").isNotNull(), split(col("Route"), "-")[0])
                       .otherwise(lit("UNKNOWN"))) \
            .withColumn("arrival_airport", 
                       when(col("Route").isNotNull(), split(col("Route"), "-")[1])
                       .otherwise(lit("UNKNOWN"))) \
            .withColumn("is_delayed", 
                       when(
                           col("scheduled_time").isNotNull() & 
                           (col("updated_time")!="--:--") & 
                           (col("updated_time") != col("scheduled_time")), 
                           "Yes"
                       ).otherwise("No")) \
            .withColumn("scheduled_departure_time",
                       concat(col("date"), lit(" "), col("scheduled_time")))
        
        if processed_df.count() > 0:
            json_df = processed_df.select(to_json(struct("*")).alias("json"))
            
            client = MongoClient('mongodb://localhost:27017/')
            db = client['flight_db']
            collection = db['flight_info']
            
            for row in json_df.collect():
                collection.insert_one(eval(row.json))
            
            print("Batch successfully processed:")
            processed_df.show(truncate=False)
        else:
            print("No valid records to process in this batch")
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        raise e

def main():
    spark = create_spark_session()
    
    kafka_df = (spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "flights")
                .option("startingOffsets", "latest")
                .load())
    
    query = kafka_df.writeStream \
        .foreachBatch(process_stream) \
        .outputMode("update") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    query.awaitTermination()

if __name__ == "__main__":
    main()