from filehandler import FileHandler
from typing import Tuple
import nltk
from tqdm import tqdm


class DketFixer():

    def __init__(self) -> None:
        self.filehandler = FileHandler(out_dir="./dketdata/")

    def logic_replace_indexes(self, text: str, logic: str) -> Tuple[str, str]:
        text = text.replace("<EOS>/<EOS>", "")
        text_tups = [nltk.tag.str2tuple(t) for t in text.split()]
        text = [t[0] for t in text_tups]

        logic = logic.replace("LOC#", "").replace("<EOS>", "")
        logic_list = logic.split()

        logic_replaced = []
        for token in logic_list:
            if token.isdigit():
                logic_replaced.append(text[int(token)])
            else:
                logic_replaced.append(token)
        return (" ".join(text), " ".join(logic_replaced))

    def prepare_dataset(self, dataset_name: str) -> None:
        training_data = self.filehandler.readlines_from_csv(
            dataset_name + ".tsv")
        text_data = []
        logic_data = []
        for text, logic in tqdm(training_data):
            td, ld = self.logic_replace_indexes(text, logic)
            text_data.append(td)
            logic_data.append(ld)

        self.filehandler.write_dataset_to_csv(
            text_data, logic_data, "dket_" + dataset_name)


df = DketFixer()
datasets = ["20k", "10k", "5k", "2k"]
for d in datasets:
    df.prepare_dataset("train_" + d)
    df.prepare_dataset("validation_" + d)
