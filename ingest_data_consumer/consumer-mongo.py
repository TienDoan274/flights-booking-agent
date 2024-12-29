from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pymongo
from pymongo import MongoClient
from pyspark.sql.functions import to_json, struct
from datetime import datetime
import json
from itertools import chain
from pyspark.sql.functions import create_map, lit

def create_spark_session():
    return (SparkSession.builder
            .appName("FlightDataProcessor")
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate())
spark = create_spark_session()
with open('iata_code/airport.json','r',encoding = 'utf-8') as f:
    airport_dict = json.load(f)

with open('iata_code/region_name.json','r',encoding = 'utf-8') as f:
    region_name_dict = json.load(f)
    
with open('iata_code/airline.json','r',encoding = 'utf-8') as f:
    airline_dict = json.load(f)
    
mapping_region_name = create_map([lit(x) for x in chain(*region_name_dict.items())])
mapping_airport = create_map([lit(x) for x in chain(*airport_dict.items())])
mapping_airline = create_map([lit(x) for x in chain(*airline_dict.items())])


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
            .withColumn("departure_airport_code", 
                       when(col("route").isNotNull(), split(col("route"), "-")[0])
                       .otherwise(lit("UNKNOWN"))) \
            .withColumn("arrival_airport_code", 
                       when(col("route").isNotNull(), split(col("route"), "-")[1])
                       .otherwise(lit("UNKNOWN"))) \
            .withColumn("is_delayed", 
                       when(
                           col("scheduled_time").isNotNull() & 
                           (col("updated_time")!="--:--") & 
                           (col("updated_time") != col("scheduled_time")), 
                           "Yes"
                       ).otherwise("No")) \
            .withColumn("departure_time",
                       concat(col("date"), lit(" "), col("scheduled_time")))\
            .withColumn("airline_code", regexp_extract(col("flight_id"), "^([A-Z]+)", 1))
        final_df = processed_df \
            .withColumn("departure_region_name", mapping_region_name[processed_df["departure_airport_code"]]) \
            .withColumn("departure_airport", mapping_airport[processed_df["departure_airport_code"]]) \
            .withColumn("arrival_region_name", mapping_region_name[processed_df["arrival_airport_code"]]) \
            .withColumn("arrival_airport", mapping_airport[processed_df["arrival_airport_code"]]) \
            .withColumn("airline",mapping_airline[processed_df["airline_code"]])\
            .drop("departure_airport_code", "arrival_airport_code", "airline_code")
                  
        if final_df.count() > 0:
            json_df = final_df.select(to_json(struct("*")).alias("json"))
            
            client = MongoClient('mongodb://localhost:27017/')
            db = client['flight_db']
            collection = db['flight_info']
            
            for row in json_df.collect():
                collection.insert_one(eval(row.json))
            
            print("Batch successfully processed:")
            final_df.show(truncate=False)
        else:
            print("No valid records to process in this batch")
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        raise e

def main():
    spark = create_spark_session()
    with open('airport.json','r',encoding = 'utf-8') as f:
        airport_dict = json.load(f)

    with open('region_name.json','r',encoding = 'utf-8') as f:
        region_name_dict = json.load(f)
        
    mapping_region_name = create_map([lit(x) for x in chain(*region_name_dict.items())])
    mapping_airport = create_map([lit(x) for x in chain(*airport_dict.items())])
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