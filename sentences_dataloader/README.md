
# Folder files


- `make_dataset.py` Download and pre-process wikipedia dump. It takes some time for executing
- `loader.py` Code for loading couples word/embedding from wikipedia dump
- `/../dataset` folder containing pre-processed dataset and index files. Automatically generated
- `ruby_postprocessing.py` Some postprocessing for files obtained using `wp2txt` 

# Generating the dataset

## Option A
run `make_dataset.py` with the following parameter:

```run(process_xml=True)```

The script will download italian dump of wikipedia and will process
xml

PROS
 - Quick and easy

CONS:
 - Processed file contains some errors


## Option B
run `make_dataset.py` with the following parameter:

```run(process_xml=False)```

The script will just download and uncompress italian dump of wikipedia.

You will need ruby (for windows users look here: https://rubyinstaller.org/downloads)

Install `wp2txt`  (https://github.com/yohasebe/wp2txt) . On ruby console type: 

```gem install wp2txt```

Once the package is installed, in the `dataset` folder :

```wp2txt -i wiki_it.xml```

After some time (~1h on my laptop) all the xml will be processed.

However, you still need removing titles (==Caratteristiche==), 
text inside brackets ([[Chiesa di Notre-Dame-de-la-Basse-Œuvre]]) 
and lines containing categories.
Also, you get lots of files.

Run `ruby_postprocessing.py` for merging all chunks and remove special lines.

PROS
 - Processed file is cleaner

CONS:
 - You need ruby
 - More steps 

