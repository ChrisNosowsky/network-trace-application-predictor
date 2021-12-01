#!/usr/bin/python
# Filename: offline-analysis-example.py
from pathlib import Path

"""
Offline analysis by replaying logs
"""

# Import MobileInsight modules
import glob
from mobile_insight.monitor import OfflineReplayer
from mobile_insight.analyzer import MsgLogger

applications = ['Airbnb', 'Amazon Shopping', 'Doordash', 'Dualingo', 'Gmail', 'Google Chrome', 'Google Maps',
                'Google Play', 'Linked in', 'Macys', 'New York Post', 'Pinterest', 'Reddit', 'Snapchat',
                'Southwest Airlines', 'Starbucks', 'Twitch', 'Twitter', 'Walmart', 'Youtube']

if __name__ == "__main__":
    home = str(Path.home())

    for app in applications:
        print("Generating XML for application: " + app)
        # get all log files
        log_files = glob.glob(home + "/MobileInsightData/" + app + "/*")
        print(log_files)
        for i, file in enumerate(log_files):

            # Initialize a monitor
            src = OfflineReplayer()
            src.set_input_path(file)
            src.enable_log_all()

            dumper = MsgLogger()  # Declare an analyzer
            dumper.set_source(src)  # Bind the analyzer to the monitor
            dumper.set_decoding(MsgLogger.XML)  # decode the message as xml
            dumper.save_decoded_msg_as(home + "/MobileInsightData/" + app + "-xml/log-" + str(i) + ".txt")
            # Start the monitoring
            src.run()
