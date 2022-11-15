import re
import os
from tqdm import tqdm

from filehandler import FileHandler


class AtomicSplitter:

    def __init__(self) -> None:
        self.filehandler = FileHandler()
        self.relations = [
            "oEffect",
            "oReact",
            "oWant",
            "xAttr",
            "xEffect",
            "xIntent",
            "xNeed",
            "xReact",
            "xWant",
        ]

    def open_closed_splitter(self, datafile_name: str) -> None:
        """
        Splits a given dataset into a closed and open dataset,
        based on if there exists an unknown word in it or not
        """
        open_data = []
        closed_data = []

        with open('./atomic/' + datafile_name, 'r') as datafile:
            raw_data = datafile.readlines()
        datafile.close()

        # TODO Add check or different thing if there data has already been
        # relation splitted, aka make these two functions compatible with each other

        header = raw_data[0]

        for data in tqdm(raw_data[1:]):
            sentence = data.split(',')[0]
            if "___" in sentence:
                open_data.append(data)
            else:
                closed_data.append(data)

        datafile_root = datafile_name.split('.')[0]

        self.filehandler.write_str_lines_to_csv(
            open_data, datafile_root, "open", header)
        self.filehandler.write_str_lines_to_csv(
            closed_data, datafile_root, "closed", header)

    """Takes a list of strings of the format "event, relation, inference"
    and returns only the lines with relations matching the given ones
    """

    def relation_splitter(self, data: list[str], relations: list[str]) -> None:
        filtered_data = []
        for d in data:
            event, relation, inference = d.split(',')
            for wanted_relation in relations:
                if relation == wanted_relation:
                    filtered_data.append([event, relation, inference])
        return filtered_data

    def parse_inferences_prefix_split(self, ips: str) -> tuple([str]):
        ips = ips.replace("\"", "")
        ips = re.split(',(?! .+\])', ips)
        return (ips[0:9], ips[9], ips[10])

    def if_then_splitter(self, data: str, include_dataset=False) -> list[str]:
        if_then_list = []
        event, *data = data.split(',', maxsplit=1)
        inference_list, _, _ = self.parse_inferences_prefix_split(
            data[0])
        for index, inferences in enumerate(inference_list):
            if inferences != "[]":
                stripped_inferences = inferences.replace(
                    '[', "").replace(']', "")
                stripped_inferences = [x.strip()
                                       for x in stripped_inferences.split(',')]
                for inference in stripped_inferences:
                    corrected_inference = self.correct_individuals(inference)
                    if_then_list.append(
                        ",".join([event, self.relations[index], corrected_inference]))

        return if_then_list

    def correct_individuals(self, inference):
        words = inference.lower().replace('.', "").split()
        corrected_inference = []
        i = 0
        while i < len(words):
            if words[i] == "person":
                if i != len(words) - 1:
                    if words[i+1] == 'x':
                        corrected_inference.append("PersonX")
                        i += 2
                        continue
                    elif words[i+1] == 'y':
                        corrected_inference.append("PersonY")
                        i += 2
                        continue
                    elif words[i+1] == 'z':
                        corrected_inference.append("PersonZ")
                        i += 2
                        continue
                    elif words[i+1] == 'z':
                        corrected_inference.append("PersonZ")
                        i += 2
                        continue
                    elif words[i+1] == "x's":
                        corrected_inference.append("PersonX's")
                        i += 2
                        continue
                    elif words[i+1] == "y's":
                        corrected_inference.append("PersonY's")
                        i += 2
                        continue
                    elif words[i+1] == "z's":
                        corrected_inference.append("PersonZ's")
                        i += 2
                        continue
            elif words[i] == "personx":
                corrected_inference.append("PersonX")
            elif words[i] == "persony":
                corrected_inference.append("PersonY")
            elif words[i] == "personz":
                corrected_inference.append("PersonZ")
            elif words[i] == 'x':
                corrected_inference.append("PersonX")
            elif words[i] == 'y':
                corrected_inference.append("PersonY")
            elif words[i] == 'z':
                corrected_inference.append("PersonZ")
            elif words[i] == "x's":
                corrected_inference.append("PersonX's")
            elif words[i] == "y's":
                corrected_inference.append("PersonY's")
            elif words[i] == "z's":
                corrected_inference.append("PersonZ's")
            else:
                corrected_inference.append(words[i])
            i += 1

        return " ".join(corrected_inference)

    def write_if_then_to_file(self, file_name: str) -> None:
        raw_data = self.filehandler.readlines_from_file(file_name)

        del raw_data[0]
        data = []

        for row in tqdm(raw_data):
            row = self.if_then_splitter(row)
            for inference in row:
                data.append(inference + '\n')
        root = file_name.split('.')[0]

        self.filehandler.write_str_lines_to_csv(data, root, "split")

    def generate_files(self, file_name: str, closed: bool = True) -> None:
        self.open_closed_splitter(file_name)
        root = file_name.split('.')[0]
        if closed:
            self.write_if_then_to_file(root + "_closed.csv")
        else:
            self.write_if_then_to_file(root + "_open.csv")

    def create_relation_dataset_from_file(self, file_name: str, relations: list[str]) -> None:
        data = self.filehandler.readlines_from_file(file_name)
        relation_data = self.relation_splitter(data, relations)
        root = file_name.split('.')[0]
        self.filehandler.write_list_lines_to_csv(
            relation_data, root, "_".join(relations))


atsp = AtomicSplitter()
# atsp.write_if_then_to_file("v4_atomic_trn_closed.csv")
atsp.create_relation_dataset_from_file(
    "v4_atomic_trn_closed_split_tagged.csv", ["xEffect", "oEffect", "xNeed", "xWant", "oWant"])
