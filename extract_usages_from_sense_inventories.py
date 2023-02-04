import os
import argparse
from collections import defaultdict
import xml
import xml.etree.ElementTree as ET
import json

# the location of sense inventories
base_dir = "/cldata/LDC/ontonotes-release-5.0/data/files/data/english/metadata/sense-inventories"


def extract_usages():
    target_files = os.listdir(base_dir)
    err_cnt = 0
    sense_dict = {}
    for tar in target_files:
        try:
            tree = ET.parse(os.path.join(base_dir, tar))
        except xml.etree.ElementTree.ParseError:
            err_cnt += 1
            continue
        root = tree.getroot()
        senses = {}
        for child in root:
            if child.tag != "sense":
                continue
            sense_number = child.attrib['n']
            sense_description = child.attrib['name']
            usages = []
            for grandchild in child:
                if grandchild.tag != "examples":
                    continue
                if grandchild.text is None:  # there are some place holder, ignore.
                    """
                    <sense group="1" n="3" name="Placeholder Sense: Do Not Choose" type="">
                        <commentary>
                            PERFORMANCE[+event]
                            The execution of an action or duty.  Often refers to a good or flawless execution of an action.
                        </commentary>
                        <examples/>
                        <mappings>
                            <wn version="2.0">Placeholder Sense</wn><omega/>
                            <pb/>
                        </mappings>
                        <SENSE_META clarity=""/>
                    </sense>
                    """
                    continue
                usages = [tmp.strip() for tmp in grandchild.text.strip().split("\n") if tmp.strip() != '']
            if len(usages) == 0:
                continue
            senses[sense_number] = [sense_description, usages]
        sense_dict[tar.replace(".xml", "")] = senses
    return sense_dict

if __name__ == "__main__":
    sense_dict = extract_usages()
    output_path = "./data/usages.json"
    with open(output_path, "w") as f:
        json.dump(sense_dict, f, indent=4)
                
    