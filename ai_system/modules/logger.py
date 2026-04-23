import csv
from datetime import datetime

def save_log(event, name="Unknown"):
    with open("database/logs.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now(), event, name])