from google.cloud import translate_v2 as translate
import tarfile
import six
import pandas as pd

def unzip(fname):
    tf = tarfile.open(fname)
    tf.extractall("data/raw")
        
def make_translations_df(german_file, english_file):
    client = translate.Client(target_language='en')
    f_de = open(german_file, encoding='utf-8')
    f_en = open(english_file, encoding='utf-8')
    translations = []
    count = 0
    for line_de, line_en in zip(f_de, f_en):
        if count % 10000 == 0:
            print('Translated {:.4f}% of lines'.format(count/1920209*100))
        if isinstance(line_de, six.binary_type):
            line_de = line_de.decode("utf-8")
        translated_eng = client.translate(line_de, source_language = 'de')
        translations.append([line_de, line_en, translated_eng["translatedText"]])
        count += 1
    f_de.close()
    f_en.close()
    df = pd.DataFrame(translations, columns = ['Original', 'Human', 'Automated'])
    return df
    
def make_datasets(data):
    shuffled = data.sample(frac=1)
    size = len(shuffled)
    train_size = int(size * 6 / 10)
    test_size = int(size * 2 / 10)
    train = shuffled.iloc[:train_size,:]
    test = shuffled.iloc[train_size:train_size+test_size,:]
    dev = shuffled.iloc[train_size+test_size:,:]
    
    train.to_csv('data/train/europarl_train.csv')
    test.to_csv('data/test/europarl_test.csv')
    dev.to_csv('data/dev/europarl_dev.csv')
        
if __name__ == "__main__":
    zipped = 'data/raw/de-en.tgz'
    german_file = 'data/raw/europarl-v7.de-en.de'
    english_file = 'data/raw/europarl-v7.de-en.en'
    # unzip(zipped)
    translated = make_translations_df(german_file, english_file)
    make_datasets(translated)