#!/usr/bin/python
# Filename: offline-analysis-example.py
import os
import sys

"""
Offline analysis by replaying logs
"""

# Import MobileInsight modules
import glob
from mobile_insight.monitor import OfflineReplayer
from mobile_insight.analyzer import MsgLogger, NrRrcAnalyzer, LteRrcAnalyzer, WcdmaRrcAnalyzer, LteNasAnalyzer, UmtsNasAnalyzer, LteMacAnalyzer, LteMeasurementAnalyzer

if __name__ == "__main__":
    # get all log files
    log_files = glob.glob("/home/hamzeh/Downloads/MobileInsightRawData/Mobile insight Data/New York Post/*")
    print(log_files)
    for i, file in enumerate(log_files):

        # Initialize a monitor
        src = OfflineReplayer()
        src.set_input_path(file)
        src.enable_log_all()

        dumper = MsgLogger()  # Declare an analyzer
        dumper.set_source(src)  # Bind the analyzer to the monitor
        dumper.set_decoding(MsgLogger.XML)  # decode the message as xml
        dumper.save_decoded_msg_as("/home/hamzeh/Desktop/mobileinsightdata/New York Post-xml/log-" + str(i) + ".txt")
        # Start the monitoring
        src.run()
