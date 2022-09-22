import os


class FileHandler:
    def __init__(self, out_dir='./generated/') -> None:
        self.out_dir = out_dir

    def readlines_from_file(self, file_path: str) -> None:
        data = []
        with open(self.out_dir + file_path, 'r') as data_file:
            data = data_file.readlines()
        return data

    def writelines_to_csv(self, data: list[str], prefix: str, suffix: str = "", header: str = None) -> None:
        data_path = self.out_dir + prefix + '_' + suffix + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w') as data_file:
            if header != None:
                data_file.writelines(header)
            for row in data:
                data_file.writelines(row)
