#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 25th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Main script to run our application
# ==============================================================================
from network import Network


if __name__ == '__main__':
    print("Running Cellular Network Trace Application...")
    path = "./data/Master_LTE_RRC_OTA_Packet_updated.xlsx"
    # path = "./data/Master_LTE_MAC_UL_Tx_Statistics.xlsx"
    n = Network()
    n.SVM_model(path)     # BEST RRC: 0.336 BEST MAC: 0.14
    # n.keras_model(path)     # BEST RRC: 0.277 BEST MAC: 0.162
    # n.linear_regression(path)   # BEST RRC: 0.060 BEST MAC: 0.00075
