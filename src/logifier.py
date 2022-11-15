import nltk

from filehandler import FileHandler


class Logifier:

    def __init__(self) -> None:
        self.filehandler = FileHandler()
        self.logic_dic = {
            "PersonX": "person(x)",
            "PersonX's": "person(x)",
            "PersonY": "person(y)",
            "PersonY's": "person(y)",
            "PersonZ": "person(z)",
            "PersonZ's": "person(z)",
        }

    def tagged_persona_inference_to_logic(self, inference: list[tuple[str]]) -> str:
        logic = []
        tagged_tups = [nltk.tag.str2tuple(t) for t in inference.split()]
        for word, _ in tagged_tups:
            logic.append(word.lower().replace('\n', ""))
        return "_".join(logic) + "(x)\n"

    def tagged_persona_event_to_logic(self, event: list[tuple[str]]) -> str:
        tagged_tupes = [nltk.tag.str2tuple(t) for t in event.split()]

        # skip examples where PersonX, PersonY and PersonZ are all included
        individuals = sum(tag == "IND" for _, tag in tagged_tupes)
        if individuals > 2:
            return (None, None)

        # set subject
        subject, _ = tagged_tupes.pop(0)

        # weird thing with quoted "PersonX is nervous"
        if subject == '"PersonX':
            return (None, None)
        else:
            logic = self.logic_dic[subject]

        # divide sentence into different parts
        verb_part = []
        verb_part_done = False
        other_person = None
        object_part = []

        while tagged_tupes:
            word, tag = tagged_tupes.pop(0)
            if tag in ["IND", "DT"]:
                verb_part_done = True
                if tag == "IND":
                    other_person = word
                if word == "no":
                    object_part.append(word)
            elif not verb_part_done:
                verb_part.append(word)
            else:
                object_part.append(word)

        # combine the verb part
        verb = "_".join(verb_part)

        if other_person != None:
            person_letter = other_person[6].lower()
            # skip adding the other person if it is personX
            if person_letter == "y":
                if object_part:
                    verb += "(x, a, y)"
                else:
                    verb += "(x, y)"
                logic += " & " + verb + " & " + self.logic_dic[other_person]
            else:
                logic += " & " + verb + "(x, a)"

        else:
            logic += " & " + verb + "(x, a)"

        if object_part:
            objectt = "_".join(object_part)
            objectt += "(a)"
            logic += " & " + objectt

        if other_person == None:
            return (logic, None)
        return (logic, other_person[6].lower())

    def persona_to_logic(self, event: list[tuple[str]], inference: list[tuple[str]]) -> str:
        event_logic, _ = self.tagged_persona_event_to_logic(event)
        if event_logic == None:
            return None
        inference_logic = self.tagged_persona_inference_to_logic(inference)
        return event_logic + " -> " + inference_logic

    def tagged_mental_inference_to_logic(self, dimension: str,  inference: list[tuple[str]], o_person: str = None) -> str:
        tagged_tupes = [nltk.tag.str2tuple(t) for t in inference.split()]
        if not tagged_tupes:
            return None

        # divide sentence into different parts
        verb_part = []
        verb_part_done = False
        other_person = None
        object_part = []

        if tagged_tupes[0][0] in ["PersonX", "PersonY", "PersonZ"]:
            word, _ = tagged_tupes.pop(0)
            other_person = word

        logic = ""

        while tagged_tupes:
            word, tag = tagged_tupes.pop(0)
            if tag in ["IND", "DT"]:
                verb_part_done = True
                if tag == "IND":
                    other_person = word
                if word == "no":
                    object_part.append(word)
            elif not verb_part_done:
                verb_part.append(word)
            else:
                object_part.append(word)

        if verb_part:
            # combine the verb part
            verb = "_".join(verb_part)
            # is it about PersonX?
            if dimension[0] == 'x':
                if other_person != None:
                    person_letter = other_person[6].lower()
                    # skip if same person
                    if person_letter == "y":
                        if object_part:
                            verb += "(x, b, y)"
                        else:
                            verb += "(x, y)"
                        logic += verb
                    else:
                        logic += verb + "(x, b)"
                else:
                    if object_part:
                        logic += verb + "(x, b)"
                    else:
                        logic += verb + "(x)"

            else:
                if o_person != None:
                    if other_person != None:
                        person_letter = other_person[6].lower()
                        # skip if same person
                        if person_letter == o_person:
                            if object_part:
                                logic += verb + '(' + o_person + ", b, " + \
                                    person_letter + ')'
                            else:
                                logic += verb + '(' + o_person + ", " + \
                                    person_letter + ')'
                        else:
                            logic += verb + "(y, x)"
                    else:
                        if object_part:
                            logic += verb + '(' + o_person + ", b)"
                        else:
                            logic += verb + '(' + o_person + ')'
                else:
                    if other_person != None:
                        person_letter = other_person[6].lower()
                        # skip if same person
                        if person_letter == o_person:
                            if object_part:
                                verb += "(u, b, " + other_person + ')'
                            else:
                                verb += "(u, " + other_person + ')'
                            logic += verb
                        else:
                            logic += verb + "(u, b)"
                    else:
                        if object_part:
                            logic += verb + "(u, b)"
                        else:
                            logic += verb + "(u)"

        if object_part:
            objectt = "_".join(object_part)
            objectt += "(b)"
            if verb_part:
                logic += " & " + objectt
            else:
                logic += objectt

        return logic + '\n'

    def mental_to_logic(self, event: list[tuple[str]], dimension: str, inference: list[tuple[str]]) -> str:
        event_logic, other_person = self.tagged_persona_event_to_logic(event)
        if event_logic == None:
            return None
        inference_logic = self.tagged_mental_inference_to_logic(
            dimension, inference, other_person)
        if inference_logic == None:
            return None
        return event_logic + " -> " + inference_logic

    def atomic_to_logic(self, dataset: list[str]) -> list[str]:
        logic = []
        for data in dataset:
            event, dimension, inference = data.split(',')
            if dimension == "xAttr":
                l = self.persona_to_logic(event, inference)
                if l != None:
                    logic.append(l)

            if dimension in ["xIntent", "xReact", "oReact"]:
                l = self.mental_to_logic(event, dimension, inference)
                if l != None:
                    logic.append(l)

            if dimension in ["xEffect", "oEffect", "xNeed", "xWant", "oWant"]:
                l = self.mental_to_logic(event, dimension, inference)
                if l != None:
                    logic.append(l)

        return logic

    def sentence_split_up(self, dataset: list[str]) -> list[str]:
        prepared_sentences = []
        for sentence in dataset:
            tagged_tups = [nltk.tag.str2tuple(t) for t in sentence.split()]
            prepared_sentence = []
            for word, _ in tagged_tups:
                prepared_sentence.append(word.lower().replace(',', " "))
            prepared_sentences.append(" ".join(prepared_sentence))
        return prepared_sentences

    def logic_split_up(self, dataset: list[str]) -> list[str]:
        split_dataset = []
        for logic in dataset:
            split_dataset.append(logic.replace('(', " (").replace('_', " "))
        return split_dataset

    def read_dataset_write_logic(self, file_name: str) -> None:
        dataset = self.filehandler.readlines_from_file(file_name)
        logic_dataset = self.atomic_to_logic(dataset)
        root = file_name.split('.')[0]
        self.filehandler.write_str_lines_to_csv(logic_dataset, root, "logic")

    def prepare_logic_dataset_from_logic(self, file_name: str) -> None:
        sentence_dataset = self.filehandler.readlines_from_file(
            file_name.replace("_logic", ""))
        prepared_sentences = self.sentence_split_up(sentence_dataset)
        logic_dataset = self.filehandler.readlines_from_file(file_name)
        prepared_logic = self.logic_split_up(logic_dataset)
        self.filehandler.write_dataset_to_csv(
            prepared_sentences, prepared_logic, "persona_dataset")


lf = Logifier()
lf.read_dataset_write_logic("v4_atomic_trn_closed_split_tagged_xAttr.csv")
lf.prepare_logic_dataset_from_logic(
    "v4_atomic_trn_closed_split_tagged_xAttr_logic.csv")
