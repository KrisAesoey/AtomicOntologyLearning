import nltk
from numpy import var
from pyrsistent import v
from sortedcollections import OrderedSet

from filehandler import FileHandler


class Logifier2:

    def __init__(self) -> None:
        self.categories = {
            "persona": ["xAttr"],
            "mental": ["xIntent", "xReact", "oReact"],
            "event": ["xEffect", "oEffect", "xNeed", "xWant", "oWant"]
        }
        self.filehandler = FileHandler("./quantified/")

    def event_to_logic(self, event: list[str]) -> list[str]:
        event_logic = []
        tagged_tupes = [nltk.tag.str2tuple(t) for t in event.split()]

        individuals = OrderedSet([word[0:7].lower()
                                  for word, tag in tagged_tupes if tag == "IND"])
        # skip examples where PersonX, PersonY and PersonZ are all included
        if len(individuals) > 3:
            return None

        # create logic of all individuals in the event and store variables
        variables = []
        for ind in individuals:
            variables.append(ind[6])
            event_logic.append("person " + "(" + ind[6] + ")")

        # find the other concepts of the event
        verb_part = []
        object_part = []
        verb_part_done = False
        tagged_tupes.pop(0)
        while tagged_tupes:
            word, tag = tagged_tupes.pop(0)
            word = word.lower()
            if tag in ["IND", "DT"]:
                verb_part_done = True
            elif verb_part and tag in ["NN", "NNS", "JJ", "JJS"]:
                verb_part_done = True
                object_part.append(word)
            elif not verb_part_done:
                verb_part.append(word)
            else:
                object_part.append(word)

        verb = " ".join(verb_part)

        if len(individuals) == 2:
            event_logic.append(verb + " (x,z,y)")
        else:
            event_logic.append(verb + " (x,z)")

        if object_part:
            obj = " ".join(object_part)
            event_logic.append(obj + " (z)")
            variables.append("z")

        return event_logic

    def inference_to_logic(self, target: str, event_logic: list[str], inference: list[str]) -> str:
        inference_logic = []
        tagged_tupes = [nltk.tag.str2tuple(t) for t in inference.split()]

        individuals = OrderedSet([word[0:7].lower()
                                  for word, tag in tagged_tupes if tag == "IND"])

        # create logic of all individuals in the inference and store variables
        variables = []
        for ind in individuals:
            variables.append(ind[6])
            ind_logic = "person " + "(" + ind[6] + ")"
            if ind_logic not in event_logic:
                inference_logic.append(ind_logic)

        verb_part = []
        verb_part_done = False
        concepts = []
        concept = []
        concept_variables = []
        variable = "a"
        for word, tag in tagged_tupes:
            word = word.lower()
            if tag in ["IND", "DT", "CC"]:
                if verb_part_done and concept:
                    # finish current concept and create new one
                    concepts.append(
                        " ".join(concept) + " (" + variable + ")")
                    concept_variables.append(variable)
                    variable = chr(ord(variable) + 1)
                    concept = []
                if verb_part:
                    verb_part_done = True

            elif not verb_part_done:
                verb_part.append(word)

            else:
                concept.append(word)
        if concept:
            concepts.append(
                " ".join(concept) + " (" + variable + ")")
            concept_variables.append(variable)

        redundant_concept = False
        # add the verb part correctly
        if verb_part:
            verb = " ".join(verb_part)
            if target == "x":
                # if no new concepts are added
                if not concepts:
                    # but both person x and y is in the inference
                    if len(individuals) == 2:
                        inference_logic.append(verb + " (x,y)")
                    else:
                        inference_logic.append(verb + " (x)")

                if concepts:
                    if len(concepts) == 1 and concepts[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (x,z,y)")
                        else:
                            inference_logic.append(verb + " (x,z)")
                    # but both person x and y is in the inference
                    elif len(individuals) == 2:
                        inference_logic.append(
                            verb + " (x," + ",".join(concept_variables) + ",y)")
                    else:
                        inference_logic.append(
                            verb + " (x," + ",".join(concept_variables) + ")")
            if target == "o":
                if "person (y)" in event_logic:
                    # if no new concepts are added
                    if not concepts:
                        # but both person x and y is in the inference
                        if len(individuals) == 2:
                            inference_logic.append(verb + " (y,x)")
                        else:
                            inference_logic.append(verb + " (y)")

                    elif len(concepts) == 1 and concepts[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (y,z,x)")
                        else:
                            inference_logic.append(verb + " (y,z)")
                    # but both person x and y is in the inference
                    elif len(individuals) == 2:
                        inference_logic.append(
                            verb + " (y," + ",".join(concept_variables) + ",x)")
                    else:
                        inference_logic.append(
                            verb + " (y," + ",".join(concept_variables) + ")")
                else:
                    # if no new concepts are added
                    if not concepts:
                        # but person x in inference
                        if len(individuals) == 1:
                            inference_logic.append(verb + " (u,x)")
                        else:
                            inference_logic.append(verb + " (u)")
                    elif len(concepts) == 1 and concepts[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (u,z,x)")
                        else:
                            inference_logic.append(verb + " (u,z)")
                    # but both person x and y is in the inference
                    elif len(individuals) == 1:
                        inference_logic.append(
                            verb + " (u," + ",".join(concept_variables) + ",x)")
                    else:
                        inference_logic.append(
                            verb + " (u," + ",".join(concept_variables) + ")")

        if not redundant_concept:
            for concept in concepts:
                if concept not in event_logic:
                    inference_logic.append(concept)
        return inference_logic

    def atomic_to_logic(self, dataset: list[str]) -> list[str]:
        logic_dataset = []
        for data in dataset:
            event, dim, inference = data.split(',')
            event_logic = self.event_to_logic(event)
            inference_logic = self.inference_to_logic(
                dim[0], event_logic, inference)

            logic = " & ".join(event_logic) + " -> " + \
                " & ".join(inference_logic)
            logic_dataset.append(logic)

        return logic_dataset

    def sentence_split_up(self, dataset: list[str]) -> list[str]:
        prepared_sentences = []
        for sentence in dataset:
            sentence = sentence.lower().replace(',', " ")
            tagged_tupes = [nltk.tag.str2tuple(t) for t in sentence.split()]
            prepared_sentence = [word for word, _ in tagged_tupes]
            prepared_sentences.append(" ".join(prepared_sentence))
        return prepared_sentences

    def read_dataset_write_logic(self, file_name: str) -> None:
        dataset = self.filehandler.readlines_from_file(file_name)
        logic_dataset = self.atomic_to_logic(dataset)
        root = file_name.split('.')[0]
        self.filehandler.write_strings_to_csv2(logic_dataset, root, "logic")

    def tagged_dataset_to_logic(self, filename, dataset_name) -> None:
        tagged_dataset = self.filehandler.readlines_from_file(
            filename.replace("_logic", ""))
        logic_dataset = self.filehandler.readlines_from_file(filename)
        untagged_sentences = self.sentence_split_up(tagged_dataset)
        assert len(untagged_sentences) == len(
            logic_dataset), f"{len(untagged_sentences)} {len(logic_dataset)}"
        self.filehandler.write_dataset_to_csv2(
            untagged_sentences, logic_dataset, dataset_name
        )


lf = Logifier2()
test = ["PersonX/IND ate/VB a/DT bird/NN,xAttr,hungry/JJ",
        "PersonX/IND finds/VBS PersonX's/IND passion/JJ,xAttr,glad/JJ",
        "PersonX/IND helps/VBS PersonY's/IND work/JJ,xAttr,helpful/JJ",
        "PersonX/IND publishes/NNS PersonY's/IND work/NN,xEffect,PersonX/IND makes/VBZ money/NN off/IN the/DT royalties/NNS",
        "PersonX/IND paints/NNS PersonX's/IND portrait/NN,xNeed,buy/NN paints/NNS and/CC material/NN",
        "PersonX/IND covers/VBZ PersonY/IND area/NN,xEffect,PersonX/IND gets/VBZ scolding/VBG from/IN PersonY/IND",
        "PersonX/IND returns/NNS to/TO PersonX's/IND work/NN,xWant,double/RB down/RP on/IN the/DT work/NN"]
test_logic = lf.atomic_to_logic(test)
for t in test_logic:
    print(t)

lf.read_dataset_write_logic("v4_atomic_trn_closed_split_tagged_xAttr.csv")
lf.read_dataset_write_logic(
    "v4_atomic_trn_closed_split_tagged_xIntent_xReact_oReact.csv")
lf.read_dataset_write_logic(
    "v4_atomic_trn_closed_split_tagged_xEffect_oEffect_xNeed_xWant_oWant.csv")
lf.read_dataset_write_logic("v4_atomic_trn_closed_split_tagged.csv")
lf.tagged_dataset_to_logic(
    "v4_atomic_trn_closed_split_tagged_xAttr_logic.csv", "persona_dataset")
lf.tagged_dataset_to_logic(
    "v4_atomic_trn_closed_split_tagged_xIntent_xReact_oReact_logic.csv", "mental_dataset")
lf.tagged_dataset_to_logic(
    "v4_atomic_trn_closed_split_tagged_xEffect_oEffect_xNeed_xWant_oWant_logic.csv", "event_dataset")
lf.tagged_dataset_to_logic(
    "v4_atomic_trn_closed_split_tagged_logic.csv", "all_dataset")
