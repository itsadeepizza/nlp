# https://github.com/tchambon/deepfrench/blob/master/tools/get-wikimedia.sh

# Download compressed wikipedia dump
# Extract- as xml file
# (Optional) Convert xml to txt

import os
from pathlib import Path


root = Path(__file__).resolve().parents[1]

lang = "it"
filename_bz2 = str(root / "dataset" / f"wiki_{lang}.bz2")
filename_xml = str(root / "dataset" / f"wiki_{lang}.xml")
processed_txt = os.path.splitext(filename_xml)[0] + "_processed.txt"

def get_wiki():
    """Download and extract data from wiki dump"""
    import bz2
    import shutil
    import wget


    wikipath = f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles.xml.bz2"

    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    if not os.path.exists(filename_bz2):
        print("Downloading wikipedia dump ...")
        wget.download(wikipath, filename_bz2)
        print("Done!")

    if not os.path.exists(filename_xml):
        print("Extracting wikipedia compressed dump ...")
        with bz2.BZ2File(filename_bz2) as fr, open(filename_xml, "wb") as fw:
            shutil.copyfileobj(fr, fw)
        print("Done!")


def normalize_text(file):
    """Convert wiki xml in plain text"""
    from tqdm import tqdm
    import re


    def extract_text(batch):
        r = re.findall("<text.*?>(.*?)</text>", batch, re.DOTALL)
        return "\n".join(r)

    def sub(chunk):
        chunk = chunk.lower()
        chunk = re.sub('\[\[file*?(.*?)\n', "", chunk) # remove file
        chunk = re.sub(r'<.*?>', "", chunk, flags=re.DOTALL) # remove xml tags
        chunk = re.sub(r'\&lt.*?\&gt;', "", chunk)
        chunk = re.sub(r'\&lt;!--.*?--\&gt;', "", chunk, flags=re.DOTALL)
        chunk = re.sub('\[\[[^]]*?\|(.*?)\]\]', "\\1", chunk) # remove link tags with text
        chunk = re.sub('\[\[([^]]*?)\]\]', "\\1", chunk) # remove link tags
        chunk = re.sub('==.*?==', "", chunk) # remove titles
        chunk = re.sub('/n *?\|.*?\n', "", chunk)
        chunk = re.sub('\[.*?\]', "", chunk) # remove website
        chunk = re.sub('\{\{.*?\}\}', "", chunk) # remove bracket tags
        # batch = re.sub(r'\{.*?\}', "", batch, re.DOTALL) # remove other tags
        for name in ["collegamenti esterni", "voci correlate", "bibliografia", 'altri progetti']:
            chunk = re.sub(fr'== {name} ==.*?\n/n\n', "", chunk, flags=re.DOTALL) # remove final
            # parts

        chunk = re.sub(r"'''", "", chunk)
        chunk = re.sub(r"''", "", chunk)
        chunk = re.sub(r"\{\{'\}\}", "'", chunk)
        # batch = re.sub(r'’', " ", batch)
        # batch = re.sub(r'′', " ", batch)
        chunk = re.sub(r'&quot;', " ", chunk)
        # batch = re.sub(r'"', " ", batch)
        # batch = re.sub(r"'", " ", batch)
        # batch = re.sub(r'“/', ' ', batch)
        chunk = re.sub(r'/n', ' ', chunk)
        chunk = re.sub(r'\&amp;', ' ', chunk)
        chunk = re.sub(r'nbsp;', ' ', chunk)
        chunk = re.sub(r'=', ' ', chunk)
        chunk = re.sub(r"\*", ' ', chunk)
        chunk = re.sub(r"\|", ' ', chunk)
        chunk = re.sub(r"categoria:", ' ', chunk)
        chunk = re.sub(r"wikipedia:", ' ', chunk)
        # batch = re.sub(r"«", ' ', batch)
        chunk = re.sub(r"{", ' ', chunk)
        chunk = re.sub(r"}", ' ', chunk)
        chunk = re.sub(r'<br />', ' ', chunk)
        chunk = re.sub(' +', ' ', chunk) # remove multiple blanks
        chunk = re.sub('\n[\n ]+', '\n', chunk) # remove multiple new lines
        chunk = re.sub('\n\d\d', '\n', chunk) # remove lines of only digits
        return chunk

    def process(batch):
        batch = extract_text(batch)
        batch = sub(batch)
        return batch


    with open(file, "r", encoding='utf-8') as source:
        with open(processed_txt, "w", encoding='utf-8') as out:
            out.write("")
        with open(processed_txt, "a", encoding='utf-8') as out:
            batch = ""
            for line in tqdm(source):
                batch += line + "/n"
                if len(batch) > 1e4:
                    out.write(process(batch))
                    batch = ""
            out.write(process(batch))


def run(process_xml=True):
    get_wiki()
    if not os.path.exists(processed_txt) and process_xml:
        print("Processing XML to text ...")
        normalize_text(filename_xml)
        print("Done!")


if __name__ == "__main__":
    run(process_xml=False)