import nltk
from nltk.stem.wordnet import WordNetLemmatizer

from tqdm import tqdm

from filehandler import FileHandler


class PreProcessor:

    def __init__(self) -> None:
        self.filehandler = FileHandler()
        self.lemmatizer = WordNetLemmatizer()
        pass

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
        return " ".join(text)

    def preprocess_sentence(self, sentence: list[tuple[str]]) -> list[tuple[str]]:
        #lemma_tags = ["NN", "VB"]
        for i, (word, tag) in enumerate(sentence):
            if tag == "CD":
                sentence[i] = tuple(["NUM", tag])
            elif word.lower() == "an" and tag == "DT":
                sentence[i] = tuple(["a", tag])
            elif word.lower() in ["personx", "persony", "personz", "x", "y", "z"]:
                sentence[i] = tuple(
                    [word[:-1].capitalize() + word[-1].capitalize(), "IND"])
            elif word.lower() in ["personx's", "persony's", "personz's"]:
                sentence[i] = tuple(
                    [word[:6].capitalize() + word[6:].capitalize(), "IND"])
            """
            elif any([lemma_tag in tag for lemma_tag in lemma_tags]):
                new_tag = lemma_tags[0] if tag in lemma_tags[0] else lemma_tags[1]
                sentence[i] = tuple(
                    [self.lemmatizer.lemmatize(word), new_tag])
            """
        return sentence

    def read_tag_write_ontology(self, file_name: str) -> None:
        data = self.filehandler.readlines_from_file(file_name)
        for i, d in tqdm(enumerate(data)):
            event, relation, inference = d.split(',')
            event_tagged = self._pos_tag(event)
            event_processed = self.preprocess_sentence(event_tagged)
            inference_tagged = self._pos_tag(inference)
            inference_processed = self.preprocess_sentence(inference_tagged)
            event_str_tuples = self.tag_tuples_to_strs(event_processed)
            event_str = self.str_list_to_str(event_str_tuples)
            inference_str_tuples = self.tag_tuples_to_strs(inference_processed)
            inference_str = self.str_list_to_str(inference_str_tuples) + '\n'
            data[i] = [event_str, relation, inference_str]

        root = file_name.split('.')[0]
        self.filehandler.write_list_lines_to_csv(data, root, "tagged")


post = PreProcessor()
post.read_tag_write_ontology("v4_atomic_trn_closed_split.csv")
