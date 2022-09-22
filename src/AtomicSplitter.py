import re
import os
from tqdm import tqdm


class AtomicSplitter:

    def __init__(self) -> None:
        self.file_directory = './generated/'
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
        pass

    def write_to_csv(self, data: list[str], prefix: str, suffix: str = "", header: str = None) -> None:
        data_path = self.file_directory + prefix + '_' + suffix + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w') as data_file:
            if header != None:
                data_file.writelines(header)
            for row in data:
                data_file.writelines(row)

        data_file.close()

    """
    Splits a given dataset into a closed and open dataset,
    based on if there exists an unknown word in it or not
    """

    def open_closed_splitter(self, datafile_name: str) -> None:
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

        self.write_to_csv(open_data, datafile_root, "open", header)
        self.write_to_csv(closed_data, datafile_root, "closed", header)

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
                    if_then_list.append(
                        ", ".join([event, self.relations[index], inference]))

        return if_then_list

    def write_if_then_to_file(self, datafile_name: str) -> None:
        data = []

        with open(self.file_directory + datafile_name, 'r') as datafile:
            raw_data = datafile.readlines()
        datafile.close()

        del raw_data[0]
        header = "event, relation, inference\n"

        for row in tqdm(raw_data):
            row = self.if_then_splitter(row)
            for inference in row:
                data.append(inference)

        data = '\n'.join(data) + '\n'

        datafile_root = datafile_name.split('.')[0]

        self.write_to_csv(data, datafile_root, "split", header)

    def generate_files(self, file_name: str, closed: bool = True) -> None:
        self.open_closed_splitter(file_name)
        root = file_name.split('.')[0]
        if closed:
            self.write_if_then_to_file(root + "_closed.csv")
        else:
            self.write_if_then_to_file(root + "_open.csv")


atsp = AtomicSplitter()

atsp.generate_files("v4_atomic_trn.csv", True)
