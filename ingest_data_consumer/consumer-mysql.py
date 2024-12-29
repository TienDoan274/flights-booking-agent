from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

def create_spark_session():
    return (SparkSession.builder
            .appName("FlightDataProcessor")
            .config("spark.jars.packages", 
                   "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,mysql:mysql-connector-java:8.0.28")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate())

def process_stream(df, epoch_id):
    """
    Xử lý mỗi batch của streaming data với việc xử lý null values
    """
    schema = StructType([
        StructField("date", StringType(), True),
        StructField("Scheduled_Time", StringType(), True),
        StructField("Updated_Time", StringType(), True),
        StructField("Route", StringType(), True),
        StructField("Flight", StringType(), True),
        StructField("Counter", StringType(), True),
        StructField("Gate", StringType(), True),
        StructField("Status", StringType(), True)
    ])
    
    try:
        # Parse JSON data
        parsed_df = df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), schema).alias("data")) \
            .select("data.*")
        
        # Filter out records with null values in primary key columns
        valid_df = parsed_df.filter(
            col("Flight").isNotNull() &
            col("date").isNotNull() &
            col("Scheduled_Time").isNotNull()
        )
        
        # Log rejected records
        rejected_df = parsed_df.subtract(valid_df)
        if rejected_df.count() > 0:
            print("Rejected records due to null values in primary key columns:")
            rejected_df.show(truncate=False)
        
        # Process valid records
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
                           col("Updated_Time").isNotNull() & 
                           col("Scheduled_Time").isNotNull() & 
                           (col("Updated_Time") != col("Scheduled_Time")), 
                           "Yes"
                       ).otherwise("No"))
        
        # Only write to MySQL if we have valid records
        if processed_df.count() > 0:
            processed_df.write \
                .format("jdbc") \
                .mode("append") \
                .option("driver", "com.mysql.cj.jdbc.Driver") \
                .option("url", "jdbc:mysql://localhost:3307/mydatabase") \
                .option("dbtable", "flight_info") \
                .option("user", "user") \
                .option("password", "password") \
                .save()
            
            print("Batch successfully processed:")
            processed_df.show(truncate=False)
        else:
            print("No valid records to process in this batch")
        
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        raise e

def main():
    spark = create_spark_session()
    
    # Đọc từ Kafka
    kafka_df = (spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")  
                .option("subscribe", "flights")
                .option("startingOffsets", "latest")
                .load())
    
    # Xử lý stream
    query = kafka_df.writeStream \
        .foreachBatch(process_stream) \
        .outputMode("update") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    query.awaitTermination()

if __name__ == "__main__":
    main()