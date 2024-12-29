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
); 