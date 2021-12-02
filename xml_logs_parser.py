import xml.etree.ElementTree as ET
import glob
import pandas as pd

def get_unique_packet_types(directory_path):
    unique_types = []
    logfiles = glob.glob(directory_path + "/*")
    target_packets = []
    for logfile in logfiles:
        with open(logfile) as file:
            # adding dummy root
            fileString = '<ROOT>{}</ROOT>'.format(file.read())
        root = ET.fromstring(fileString)
        packets = root.findall("dm_log_packet")
        for packet in packets:
            for children in packet:
                if children.get("key") == "type_id":
                    if children.text not in unique_types:
                        unique_types.append(children.text)
    return unique_types
# provide path for the folder than contains all the logs for a single app at a time, otherwise we get data mixed up.
def get_packets_with_type(packet_type, directory_path):
    logfiles = glob.glob(directory_path + "/*")
    target_packets = []
    for logfile in logfiles:
        with open(logfile) as file:
            # adding dummy root
            fileString = '<ROOT>{}</ROOT>'.format(file.read())
        root = ET.fromstring(fileString)
        packets = root.findall("dm_log_packet")
        for packet in packets:
            for children in packet:
                if children.get("key") == "type_id":
                    if children.text == packet_type:
                        target_packets.append(packet)
    return target_packets

# the assumption for this to work is that all packets of same type have same structure and elements, so that if we get
# the key for every element we will have that key for every packet in the same place ( column in the csv )
# ( same type packets have same exact tags but different data) check packet_sample file and you will understand code below

def convert_xml_packets_to_csv_rows(packets, output_path):
    rows = []
    columns = []
    columns_written = False
    for p in packets:
        row = []
        tree = ET.ElementTree(element= p)
        for element in tree.iter():
            if element.get("key") != "Subpackets" and element.get("key") != "Sample" and element.get("key") is not None:
                row.append(element.text)
                if not columns_written:
                    columns.append(element.get("key"))
            # #file.write(ET.tostring(p, encoding='unicode', method='xml'))  # for visualization purposes
            # for el in p:
            #     # this element has children
            #     if el.get("type") == "list":
            #         for list in el:
            #             for item in list:
            #                 for dict in item:
            #                     for pair in dict:
            #                             row.append(pair.text)
            #                             if pair.get("key") not in columns:
            #                                 columns.append(pair.get("key"))
            #
            #     else:
            #         # direct children to dm_log_packet
            #         row.append(el.text)
            #         if el.get("key") not in columns:
            #             columns.append(el.get("key"))
        rows.append(row)
        # go over all possible columns of the first packet, assuming all packets of the same type have same structure
        columns_written = True
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path +'.csv', index=False)
failed_packet_types = []
unique_types = get_unique_packet_types("./Mobile-Insight-XML-data/Twitter-xml")
for packet_type in unique_types:
    packets = get_packets_with_type(packet_type, "./Mobile-Insight-XML-data/Twitter-xml")
    try:

        convert_xml_packets_to_csv_rows(packets, "./data/succeed/Twitter/packets for " + str(packet_type))
    except:
        failed_packet_types.append(packet_type)
        print("converstion to csv failed for packet type: " + packet_type)