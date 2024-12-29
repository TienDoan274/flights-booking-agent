import mysql.connector
from mysql.connector import Error

def create_flight_table():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            port="3307",
            user="user",     
            password="password", 
            database="mydatabase"  
        )

        cursor = connection.cursor()

        create_table_query = """
        CREATE TABLE flight_info (
            date VARCHAR(255),
            Scheduled_Time VARCHAR(10),
            Updated_Time VARCHAR(10),
            Route VARCHAR(255),
            Flight VARCHAR(50),
            Counter VARCHAR(50),
            Gate VARCHAR(20),
            Status VARCHAR(10),
            processing_timestamp TIMESTAMP,
            departure_airport VARCHAR(10),
            arrival_airport VARCHAR(10),
            is_delayed VARCHAR(5),
            PRIMARY KEY (Flight, date, Scheduled_Time)
        )
        """

        cursor.execute(create_table_query)
        connection.commit()
        print("Flight info table created successfully!")

    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    create_flight_table()