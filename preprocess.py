#!/usr/bin/env python3
# ==============================================================================
# CSE 824
# Created on: November 28th, 2021
# Authors: Chris Nosowsky (nosowsky@msu.edu),
#          Hamzeh Alzweri (alzwerih@msu.edu)
#
# Preprocess the xml files into input ready format
# for our machine learning classifier
# ==============================================================================

import xml.etree.ElementTree as ET


class PreProcess:
    def __init__(self):
        pass

    def parse_xml(self, file):
        tree = ET.parse(file)
        root = tree.getroot()
