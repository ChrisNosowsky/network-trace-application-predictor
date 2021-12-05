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
    # path = "./data/Master_LTE_PDCP_DL_Stats.xlsx"
    path = "./data/Master_LTE_PDCP_UL_Stats.xlsx"
    # path = "./data/Master_LTE_RRC_OTA_Packet_updated.xlsx"
    # path = "./data/Master_LTE_MAC_UL_Tx_Statistics.xlsx"
    # path = "./data/Master_LTE_MAC_Rach_Attempt.xlsx"                  # NOT ABLE TO WORK RN
    # path = "./data/Master_LTE_MAC_Rach_Trigger.xlsx"                  # NOT ABLE TO WORK RN
    # path = "./data/Master_LTE_NAS_EMM_State.xlsx"                     # NOT ABLE TO WORK RN NEED MORE DATA
    # path = "./data/Master_LTE_NAS_ESM_State.xlsx"                     # NOT ABLE TO WORK RN NEED MORE DATA
    # path = "./data/Master_LTE_PDCP_DL_SRB_Integrity_Data_PDU.xlsx"    # NOT ABLE TO WORK RN NEED MORE DATA
    # path = "./data/Master_LTE_RLC_DL_Stats.xlsx"                      # NOT ABLE TO WORK RN NEED MORE DATA
    # path = "./data/Master_LTE_RLC_UL_Stats.xlsx"

    # Please see all_accuracy.xlsx for list of accuracies I got
    # Best one I got: LTE_PDCP_UL_Stats
    n = Network()
    print("========LOGISTIC REGRESSION========")
    n.logistic_regression(path)
    print("========SVM========")
    n.SVM_model(path)
    print("========KERAS========")
    n.keras_model(path)
    print("========LINEAR REGRESSION========")
    n.linear_regression(path)
    print("========XG BOOST========")
    n.xg_boost(path)
