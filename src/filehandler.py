import os
import csv


class FileHandler:
    def __init__(self, out_dir='./generated/') -> None:
        self.out_dir = out_dir

    def readlines_from_file(self, file_path: str) -> list[str]:
        data = []
        with open(self.out_dir + file_path, 'r') as data_file:
            data = data_file.readlines()
        return data

    def readlines_from_csv(self, file_path: str) -> list[str]:
        data = []
        with open(self.out_dir + file_path) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                data.append(line)
        return data

    def write_str_lines_to_csv(self, data: list[str], prefix: str, suffix: str = "", header: str = None) -> None:
        data_path = self.out_dir + prefix + '_' + suffix + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w') as data_file:
            if header != None:
                data_file.writelines(header)
            for row in data:
                data_file.writelines(row)

    def write_list_lines_to_csv(self, data: list[list[str]], prefix: str, suffix: str = "", header: str = None) -> None:
        data_path = self.out_dir + prefix + '_' + suffix + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w') as data_file:
            if header != None:
                data_file.writelines(header)
            for row in data:
                row = ",".join(row)
                data_file.writelines(row)

    def write_dataset_to_csv(self, contexts: list[str], targets: list[str], name: str) -> None:
        data_path = self.out_dir + "/datasets/" + name + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\t')
            for c, t in zip(contexts, targets):
                writer.writerow([c.replace('\n', ""), t.replace('\n', "")])

    def write_strings_to_csv2(self, data: list[str], prefix: str, suffix: str) -> None:
        data_path = self.out_dir + prefix + '_' + suffix + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\n')
            writer.writerow(data)

    def write_dataset_to_csv2(self, contexts: list[str], targets: list[str], name: str) -> None:
        data_path = self.out_dir + "/datasets/" + name + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\t')
            for c, t in zip(contexts, targets):
                writer.writerow([c, t])
