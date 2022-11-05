# https://github.com/tchambon/deepfrench/blob/master/tools/get-wikimedia.sh

import os

lang = "it"
filename_bz2 = f"dataset/wiki_{lang}.bz2"
filename_xml = f"dataset/wiki_{lang}.xml"
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

    def sub(batch):
        batch = batch.lower()
        batch = re.sub('\[\[file*?(.*?)\n', "", batch) # remove file
        batch = re.sub(r'<.*?>', "", batch, flags=re.DOTALL) # remove xml tags
        batch = re.sub(r'\&lt.*?\&gt;', "", batch)
        batch = re.sub(r'\&lt;!--.*?--\&gt;', "", batch, flags=re.DOTALL)
        batch = re.sub('\[\[[^]]*?\|(.*?)\]\]', "\\1", batch) # remove link tags with text
        batch = re.sub('\[\[([^]]*?)\]\]', "\\1", batch) # remove link tags
        batch = re.sub('==.*?==', "", batch) # remove titles
        batch = re.sub('/n *?\|.*?\n', "", batch)
        batch = re.sub('\[.*?\]', "", batch) # remove website
        batch = re.sub('\{\{.*?\}\}', "", batch) # remove bracket tags
        # batch = re.sub(r'\{.*?\}', "", batch, re.DOTALL) # remove other tags
        for name in ["collegamenti esterni", "voci correlate", "bibliografia", 'altri progetti']:
            batch = re.sub(fr'== {name} ==.*?\n/n\n', "", batch, flags=re.DOTALL) # remove final
            # parts

        batch = re.sub(r"'''", "", batch)
        batch = re.sub(r"''", "", batch)
        batch = re.sub(r"\{\{'\}\}", "'", batch)
        # batch = re.sub(r'’', " ", batch)
        # batch = re.sub(r'′', " ", batch)
        batch = re.sub(r'&quot;', " ", batch)
        # batch = re.sub(r'"', " ", batch)
        # batch = re.sub(r"'", " ", batch)
        # batch = re.sub(r'“/', ' ', batch)
        batch = re.sub(r'/n', ' ', batch)
        batch = re.sub(r'\&amp;', ' ', batch)
        batch = re.sub(r'nbsp;', ' ', batch)
        batch = re.sub(r'=', ' ', batch)
        batch = re.sub(r"\*", ' ', batch)
        batch = re.sub(r"\|", ' ', batch)
        batch = re.sub(r"categoria:", ' ', batch)
        batch = re.sub(r"wikipedia:", ' ', batch)
        # batch = re.sub(r"«", ' ', batch)
        batch = re.sub(r"{", ' ', batch)
        batch = re.sub(r"}", ' ', batch)
        batch = re.sub(r'<br />', ' ', batch)
        batch = re.sub(' +', ' ', batch) # remove multiple blanks
        batch = re.sub('\n[\n ]+', '\n', batch) # remove multiple new lines
        batch = re.sub('\n\d\d', '\n', batch) # remove lines of only digits
        return batch

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


# sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
#     -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
#     -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
#     -e 's/«/ /g' | tr 0-9 " "

def make_vocab():
    import pickle
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(input='filename', max_features=15000)
    X = vectorizer.fit([processed_txt])
    with open("dataset/vectorizer.pickle", 'wb') as f:
        pickle.dump(vectorizer, f)


def run():
    get_wiki()
    if not os.path.exists(processed_txt):
        print("Processing XML to text ...")
        normalize_text(filename_xml)
        print("Done!")
    if not os.path.exists("dataset/vectorizer.pickle"):
        print("Listing words ...")
        make_vocab(filename_xml)
        print("Done!")


if __name__ == "__main__":
    run()