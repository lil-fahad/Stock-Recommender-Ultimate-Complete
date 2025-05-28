import schedule
import time
import subprocess

def retrain():
    print("Running retraining script...")
    subprocess.run(["python", "trainer.py"])

schedule.every().day.at("02:00").do(retrain)

while True:
    schedule.run_pending()
    time.sleep(60)