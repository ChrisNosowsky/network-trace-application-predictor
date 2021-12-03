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
    print(len(packets))
    packets = [packets[0]]
    firstInnerItem = True
    for p in packets:
        row = []
        tree = ET.ElementTree(element= p)
        for element in tree.iter():
            if element.get("key") != "Subpackets" and element.get("key") is not None and firstInnerItem:
                row.append(element.text)
                if element.get("key") is not None and element.get("key") not in columns:
                    columns.append(element.get("key"))
            elif element.get("key") == "Subpackets" and element.get("key") is not None:
                subpackets = []
                subpacket = dict()
                for e in element.iter():
                    if e.tag == "item" and 'type' in e.attrib:
                        if firstInnerItem:
                            firstInnerItem = False
                        else:
                            subpackets.append(subpacket)
                            subpacket = dict()

                    if e.tag == "pair" and element.get("key") is not None and e.text is not None:
                        try:
                            val = float(e.text)
                        except ValueError:
                            if e.text.isdigit():
                                val = int(e.text)
                            else:
                                val = e.text
                        subpacket[e.get("key")] = val
                subpackets.append(subpacket)
                row.append(subpackets)
                if element.get("key") is not None and element.get("key") not in columns:
                    columns.append(element.get("key"))
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path +'.csv', index=False)


failed_packet_types = []
unique_types = get_unique_packet_types("./data")
print(unique_types)
unique_types = [unique_types[0]]    # Do next one by moving up index. Done first one so far.
print(unique_types)
for packet_type in unique_types:
    print(packet_type)
    packets = get_packets_with_type(packet_type, "./data")
    try:
        convert_xml_packets_to_csv_rows(packets, "packets_for_" + str(packet_type))
    except:
        failed_packet_types.append(packet_type)
        print("converstion to csv failed for packet type: " + packet_type)