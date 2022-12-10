# Execute this script after wp2txt
from make_dataset import *
import os
import glob
from tqdm import tqdm
import re


def process_chunk(chunk):
    chunk = re.sub('\[.*?\].*', "", chunk)  # remove brackets
    chunk = re.sub('==.*?==.*', "", chunk)  # remove titles
    chunk = re.sub('CATEGORIES:.*', "", chunk)  # remove CATEGORIES
    chunk += "\n"
    chunk = re.sub('\r', ' ', chunk)
    chunk = re.sub(' +', ' ', chunk)  # remove multiple blanks
    chunk = re.sub('\n[\n ]+', '\n', chunk)  # remove multiple new lines
    return chunk


# CHECK IF RUBY HAS BEEN EXECUTED
base = os.path.splitext(filename_xml)[0]
processed_ruby_txt = base + "_ruby_processed.txt"
if os.path.exists(processed_ruby_txt):
    raise RuntimeError("A processed file exists already, delete it before executing the script")
# Make a list of all .txt and .xml files
txt_files = glob.glob(f"{base}-*.txt")
xml_files = glob.glob(f"{base}-*.xml")
if len(txt_files) > 1 and len(txt_files) >= len(xml_files):
    print("Ok, some ruby has already been executed here ...")
else:
    raise RuntimeError("Run wp2vec before ! (check th README)")
# Remove xml files
for xml in xml_files:
    os.remove(xml)
# Create empty big text file
with open(processed_ruby_txt, "w", encoding='utf-8') as f:
    f.write("")
# Append ruby-processed files, removing special text
with open(processed_ruby_txt, "a", encoding='utf-8') as f:
    for txt in tqdm(txt_files[:]):
        with open(txt, "r", encoding='utf-8') as t:
            chunk = process_chunk(t.read())
        f.write(chunk)
# Remove little txt files
for txt in txt_files:
    os.remove(txt)