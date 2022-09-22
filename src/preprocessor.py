import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm

from filehandler import FileHandler


class PreProcessor:

    def __init__(self) -> None:
        self.filehandler = FileHandler()
        self.lemmatizer = WordNetLemmatizer()
        pass

    def lemmatize_words(self, data: list[tuple[str]]) -> list[str]:
        for i, words in tqdm(enumerate(data)):
            data[i] = " ".join(self.lemmatizer.lemmatize(word)
                               for word in words.split())
        return data

    def _pos_tag(self, inference: str):
        inference = inference.replace(',', "").split()
        tagged = nltk.pos_tag(inference)
        return tagged

    def pos_tag_list(self, data: list[str]) -> list[str]:
        for i, d in tqdm(enumerate(data)):
            data[i] = self._pos_tag(d)
        return data

    """
    Takes a standard pos-tagged tuple (word, tag) and transforms it
    to the form of word/tag. ex: (bike, NN) -> bike/NN
    """

    def tag_tuple_to_str(self, tag_tuple: tuple[str]) -> str:
        return tag_tuple[0] + "/" + tag_tuple[1]

    def tag_tuples_to_strs(self, tag_tuples: list[tuple[str]]) -> list[str]:
        return [self.tag_tuple_to_str(t) for t in tag_tuples]

    def str_list_to_str(self, text: list[str]) -> str:
        return " ".join(text) + " ./. <EOS>/<EOS>\n"

    def preprocess(self, data: list[tuple[str]]) -> list[tuple[str]]:
        lemma_tags = ["NN", "VB"]
        for words in tqdm(data):
            for i, (word, tag) in enumerate(words):
                if tag == "CD":
                    words[i] = tuple(["NUM", tag])
                elif word.lower() == "an" and tag == "DT":
                    words[i] = tuple(["a", tag])
                elif any([lemma_tag in tag for lemma_tag in lemma_tags]):
                    new_tag = lemma_tags[0] if tag in lemma_tags[0] else lemma_tags[1]
                    words[i] = tuple(
                        [self.lemmatizer.lemmatize(word), new_tag])
        return data

    def read_tag_write(self, file_name: str, lemmatize: bool = False) -> None:
        data = self.filehandler.readlines_from_file(file_name)
        data = self.pos_tag_list(data)
        data = self.preprocess(data)

        for i, d in tqdm(enumerate(data)):
            data[i] = self.tag_tuples_to_strs(d)

        for i, d in tqdm(enumerate(data)):
            data[i] = self.str_list_to_str(d)
        self.filehandler.writelines_to_csv(data, file_name, "tagged")


post = PreProcessor()
post.read_tag_write("v4_atomic_trn_closed_split.csv", True)
