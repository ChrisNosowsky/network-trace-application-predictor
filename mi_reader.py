#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 28th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Convert and listen for mi2log files containing our cellular traces
# into XML format
# ==============================================================================

from mobile_insight.monitor import OfflineReplayer, OnlineMonitor
from mobile_insight.analyzer import LteRrcAnalyzer, WcdmaRrcAnalyzer, NrRrcAnalyzer, MsgLogger


class MIReader:
    def __init__(self):
        pass

    @staticmethod
    def offline_analyzer():
        src = OfflineReplayer()
        # Load offline logs
        src.set_input_path("./offline_log_examples/")

        # RRC analyzer

        nr_rrc_analyzer = NrRrcAnalyzer()  # 5G NR
        nr_rrc_analyzer.set_source(src)  # bind with the monitor

        lte_rrc_analyzer = LteRrcAnalyzer()  # 4G LTE
        lte_rrc_analyzer.set_source(src)  # bind with the monitor

        wcdma_rrc_analyzer = WcdmaRrcAnalyzer()  # 3G WCDMA
        wcdma_rrc_analyzer.set_source(src)  # bind with the monitor

        dumper = MsgLogger()  # Declare an analyzer
        dumper.set_source(src)  # Bind the analyzer to the monitor
        dumper.set_decoding(MsgLogger.XML)  # decode the message as xml
        dumper.save_decoded_msg_as('./xml_logs/test.xml')
        src.run()

    @staticmethod
    def online_analyzer(port, rate):
        # Initialize a 3G/4G monitor
        src = OnlineMonitor()
        src.set_serial_port(port)  # the serial port to collect the traces
        src.set_baudrate(rate)  # the baudrate of the port

        # Specify logs to be collected: RRC (radio resource control) in this example
        src.enable_log("5G_NR_RRC_OTA_Packet")  # 5G RRC
        src.enable_log("LTE_RRC_OTA_Packet")  # 4G LTE RRC
        src.enable_log("WCDMA_RRC_OTA_Packet")  # 3G WCDMA RRC

        # Save the monitoring results as an offline log
        src.save_log_as("./offline_log_examples/monitor-example.mi2log")

        # Print messages.
        dumper = MsgLogger()  # Declare an analyzer
        dumper.set_source(src)  # Bind the analyzer to the monitor
        dumper.set_decoding(MsgLogger.XML)  # decode the message as xml

        # Start the monitoring
        src.run()