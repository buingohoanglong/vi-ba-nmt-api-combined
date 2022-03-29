import string

from transformers import AutoTokenizer, AutoModel

from common.model_types import ModelTypes
from config.config import Configuration
from model.dictionary_translate.translate import Translator
from services.vn_core_service import VnCoreService
from pipeline.transformer_pgn_translate import TransformerPGNTranslator
from pipeline.bart_pho_translate import BartPhoTranslator


class TranslationPipeline:
    def __init__(self, config: Configuration):
        self.config = config
        self.vn_core_service = VnCoreService(config)
        self.bart_pho_model = BartPhoTranslator(config)
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.loan_former_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                          ModelTypes.LOAN_FORMER)
        self.pho_bert_fused_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                             ModelTypes.PHOBERT_FUSED)
        self.transformer_model = TransformerPGNTranslator(config, tokenizer, self.vn_core_service, pho_bert,
                                                          ModelTypes.TRANSFORMER)
        self.dictionary_translator = Translator(config, self.vn_core_service)

    @staticmethod
    def preprocess(text):
        for c in string.punctuation:
            text = text.replace(c, f" {c} ")
        return text

    @staticmethod
    def add_dot(text):
        text = text.strip()
        if text[-1] not in ".:?!":
            text += " ."
        return text

    @staticmethod
    def drop_punctuation(text):
        for c in string.punctuation:
            text = text.replace(c, "")
        text = text.strip()
        return text

    def count_words(self, text, ners):
        for ner in ners:
            text = text.replace(ner, "")
        text = self.drop_punctuation(text)
        return len(text.split())

    def translate_sent(self, text, model=ModelTypes.COMBINED):
        # if model == ModelTypes.TRANSFORMER:
        #     text = self.preprocess(text)
        #     translated = None
        #     ners = self.vn_core_service.get_ner(text)
        #     num_words = self.count_words(text, ners)
        #     if num_words <= 7:
        #         translated = self.dictionary_translator.translate_word(text, ners)
        #     if translated is None:
        #         # print("Model Translate")
        #         text = self.add_dot(text)
        #         if len(ners) > 0:
        #             print("HAS NER")
        #             print("LoanFormer")
        #             return self.loan_former_model([text])[0]
        #         elif num_words <= 7:
        #             print("TRANSFORMER")
        #             return self.transformer_model([text])[0]
        #         else:
        #             print("BARTPhoModel")
        #             return self.bart_pho_model(text)
        #     else:
        #         print("Dictionary Translate")
        #         return translated
        # elif model == ModelTypes.LOAN_FORMER:
        #     return self.loan_former_model([text])[0]
        # else:
        if model == ModelTypes.TRANSFORMER:
            print("TRANSFORMER")
            return self.transformer_model([text])[0]
        elif model == ModelTypes.PHOBERT_FUSED:
            print("PhoBERT-fused NMT")
            return self.pho_bert_fused_model([text])[0]
        elif model == ModelTypes.LOAN_FORMER:
            print("LoanFormer")
            return self.loan_former_model([text])[0]        
        else:  
            print("BARTPHO + DICT")
            #BART PHO + DICTIONARY ONLY
            text = self.preprocess(text)
            translated = None
            ners = self.vn_core_service.get_ner(text)
            num_words = self.count_words(text, ners)
            if num_words <= 7:
                translated = self.dictionary_translator.translate_word(text, ners)
            if translated is None:
                # print("Model Translate")
                text = self.add_dot(text)
                return self.bart_pho_model(text)
            else:
                print("Dictionary Translate")
                return translated

    async def __call__(self, text, model=ModelTypes.COMBINED):
        vi_paragraphs = text.split('\n')
        ba_paragraphs = []
        for paragraph in vi_paragraphs:
            sents = self.vn_core_service.tokenize(paragraph)
            sents = [" ".join(sent).replace("_", " ") for sent in sents]
            translated_sentences = [self.translate_sent(sent, model) for sent in sents]
            translated_paragraph = " ".join(translated_sentences), ""
            ba_paragraphs.append(translated_paragraph if paragraph != '' else '')
        return ba_paragraphs


if __name__ == "__main__":
    config = Configuration()
    pipeline = TranslationPipeline(config)
    output = pipeline("tôi là sinh viên trường đại học bach khoa.")
    print(output)


