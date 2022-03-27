import os
import re
import string
from tqdm import tqdm

from config.config import Configuration
from services.vn_core_service import VnCoreService


class Translator:
    def __init__(self, config: Configuration, vn_core_service: VnCoreService):
        self.config = config
        self.punctuation = string.punctuation + "–"
        self.dictionary = self.load_dictionary(config.vi_ba_dictionary_path, config.vi_ba_train_data_folder)
        # self.annotator = VnCoreNLP(address=vn_core_host, port=vn_core_port)
        vi_words = list(self.dictionary.keys())
        vi_words = [f"{w}" for w in vi_words]
        vi_words.sort(key=lambda w: len(w), reverse=True)
        self.re = re.compile("|".join(vi_words))
        self.vn_core_service = vn_core_service

    def load_dictionary(self, dictionary_path, train_data_folder):
        def get_text(item):
            item = item.replace("\n", "").strip().replace("(", " ").replace(")", " ")
            while "  " in item:
                item = item.replace("  ", " ")
            return item.strip()

        def read_extra_data(data_folder):
            vi_paths = [os.path.join(data_folder, f"{mode}.vi") for mode in ["train", "valid"]]
            ba_paths = [os.path.join(data_folder, f"{mode}.ba") for mode in ["train", "valid"]]
            extra_dict = {}
            for vi_path, ba_path in zip(vi_paths, ba_paths):
                for vi_line, ba_line in zip(open(vi_path, "r", encoding="utf8").readlines(),
                                            open(ba_path, "r", encoding="utf8").readlines()):
                    for c in self.punctuation:
                        vi_line = vi_line.replace(c, f" ")
                        ba_line = ba_line.replace(c, f" ")
                    while "  " in vi_line:
                        vi_line = vi_line.replace("  ", " ")
                    while "  " in ba_line:
                        ba_line = ba_line.replace("  ", " ")
                    vi_line = vi_line.rstrip()
                    ba_line = ba_line.rstrip()
                    if len(vi_line.split()) <= 4:
                        extra_dict[vi_line] = ba_line
            return extra_dict

        files = os.listdir(dictionary_path)
        vi_file = os.path.join(dictionary_path, [item for item in files if "vi" in item][0])
        ba_file = os.path.join(dictionary_path, [item for item in files if "bana" in item][0])
        vi_data = [get_text(item) for item in open(vi_file, "r", encoding="utf8").readlines()]
        ba_data = [get_text(item) for item in open(ba_file, "r", encoding="utf8").readlines()]
        dictionary = {}
        for vi, ba in zip(vi_data, ba_data):
            if len(vi) == 0:
                continue
            dictionary[vi] = ba
            up_vi = vi[0].upper() + vi[1:]
            up_ba = ba[0].upper() + ba[1:]
            dictionary[up_vi] = up_ba

        extra_dictionary = read_extra_data(train_data_folder)
        for vi, ba in extra_dictionary.items():
            if vi not in dictionary:
                dictionary[vi] = ba
                up_vi = vi[0].upper() + vi[1:]
                up_ba = ba[0].upper() + ba[1:]
                dictionary[up_vi] = up_ba
        return dictionary

    def __call__(self, text):
        sentences = self.annotator.tokenize(text)
        out = []
        for words in sentences:
            sentence_out = []
            for word in words:
                word = word.replace("_", " ")
                sentence_out.append(self.dictionary.get(word.lower(), word))
            sentence = " ".join(sentence_out)
            out.append(sentence)
        return ". ".join(out)

    def translate_word_(self, word):
        output = self.dictionary.get(word.lower())
        if output is None and ("(" in word or ")" in word):
            word = word.replace("(", " ").replace(")", " ")
            while "  " in word:
                word = word.replace("  ", " ")
            output = self.dictionary.get(word.lower())
        if output is None:
            words = word.split()
            if len(words) > 1:
                output = []
                for w in words:
                    translate_word = self.dictionary.get(w.lower())
                    if translate_word is None:
                        output = None
                        break
                    else:
                        output.append(translate_word)
                if output is not None:
                    output = " ".join(output)
        return output

    def _translate_word(self, word, ners, replace_all=True):
        word = f" {word} "
        for c in self.punctuation:
            word = word.replace(c, f" {c} ")

        words = list(self.re.findall(word))
        words = [item for item in words if f" {item.strip()} " in word]
        word_ = word
        for w in words:
            word_ = word_.replace(w, "")
            word = word.replace(w, self.dictionary.get(w.strip(), w.strip()))
        for c in self.punctuation:
            word_ = word_.replace(c, "")

        for ner in ners:
            word_ = word_.replace(ner, "")

        if len(word_.strip()) > 0 and replace_all:
            return None
        else:
            while "  " in word:
                word = word.replace("  ", " ")
            word = word.strip()
        return word

    def translate_word(self, word, ners=None, replace_all=True):
        if ners is None:
            ners = self.vn_core_service.get_ner(word)
        output = self._translate_word(word, ners, replace_all)
        if output is None:
            for c in self.punctuation:
                word = word.replace(c, " ")
            while "  " in word:
                word = word.replace("  ", " ")
            word = word.strip()
            output = self._translate_word(word, ners)

        return output

    # print(words)


def main():
    config = Configuration()
    vn_core_nlp = VnCoreService(config)
    translator = Translator(config, vn_core_service=vn_core_nlp)
    # data_path = "transformer-bert-pgn/data/vi-ba/test.vi"
    src_path = "translated_data/loanformer.txt"
    src_data = open(src_path, "r", encoding="utf8").readlines()
    src_data = [": ".join(item.rstrip().split(": ")[1:]) for item in src_data if item[:4] == "--|S"]
    c = 0
    for i, line in enumerate(tqdm(src_data)):
        line = line.rstrip()
        # ners = vn_core_nlp.get_ner(line)
        ners = []
        output = translator.translate_word(line, ners, replace_all=False)
        with open("translated_data/dict_translate.txt", "a", encoding="utf8") as f:
            f.write(f"--|P-{i}: {output}\n")
        if output is None:
            c += 1
            print(f"|{line}|")
    print(translator.translate_word("những con trâu (ấy)"))
    print(c)


if __name__ == "__main__":
    main()
