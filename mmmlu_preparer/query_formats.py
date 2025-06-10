import re
import json
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Union
import numpy as np
from datasets import Dataset
from .read_mmmlu_dataset import MMMLULanguage



# The original base prompt
# common.py
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# Base prompt with variable output format
BASE_TEMPLATE_FREE = """
Answer the following multiple choice question. {Output_format} Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

# The sentecnes of base prompt
BASE_TASK = "Answer the following multiple choice question."
BASE_OUTPUT_FORMAT = "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."
BASE_INSTRUCTION = "Think step by step before answering."


JSON_ANSWER_OUTPUT_FORMAT = """The last lines of your response should be of the following format:

{
    "Answer": "<$LETTER>"
}

where LETTER must be one of ABCD. Output your answer in the valid JSON object. Ensure the output is valid and parseable as JSON.""".strip()

JSON_FULL_OUTPUT_FORMAT = """Your response should be of the following format:

{
    "Reasoning": "<Your step-by-step reasoning, written in one line>",
    "Answer": "<$LETTER>"
}

where LETTER is one of ABCD. Only output the valid JSON object, and nothing else. Ensure the output is valid and parseable as JSON.""".strip()

XML_ANSWER_OUTPUT_FORMAT = """The last lines of your response should be of the following format:

<root>
    <Answer>$LETTER</Answer>
</root>

where LETTER is one of ABCD. Output your answer in the valid XML element. Ensure the output is valid and parseable as XML.
""".strip()

XML_FULL_OUTPUT_FORMAT = """Your response should be of the following format:

<root>
    <Reasoning>[Your step-by-step reasoning, written in one line]</Reasoning>
    <Answer>[$LETTER]</Answer>
</root>
where LETTER is one of ABCD. Only output the valid XML object, and nothing else. Ensure the output is valid and parseable as XML.
""".strip()

DEFAULT_LETTER_ORDER = np.array(["A", "B", "C", "D"])
DEFAULT_ORDER = np.arange(0, len(DEFAULT_LETTER_ORDER))

def get_json_input(common_query_filling: dict, options_filling: dict) -> dict:
    # Draft Json. Not confirm
    json_dict = {
        "Task": BASE_TASK,
        "Output_format": "",
        "Instruction": BASE_INSTRUCTION,
        "Question": "",
        "Options": {
            "A": "",
            "B": "",
            "C": "",
            "D": ""
        }
    }

    json_dict['Output_format'] = common_query_filling['Output_format']
    json_dict['Question'] = common_query_filling['Question']

    for opt_id in DEFAULT_LETTER_ORDER:
        opt_id = opt_id.upper()
        json_dict['Options'][opt_id] = options_filling[opt_id]
    return json_dict

def get_xml_input(common_query_filling: dict, options_filling: dict) -> str:
    '''
    common_query_filling - question and output_format
    options_filling - options and their contents
    '''
    xml_root = ET.Element("root")
    task_tag = ET.SubElement(xml_root, 'Task')
    task_tag.text = BASE_TASK

    output_tag = ET.SubElement(xml_root, "Output_format")
    output_tag.text = common_query_filling['Output_format']

    instruction_tag = ET.SubElement(xml_root, "Instruction")
    instruction_tag.text = BASE_INSTRUCTION

    question_tag = ET.SubElement(xml_root, "Question")
    question_tag.text = common_query_filling['Question']

    options_tag = ET.SubElement(xml_root, "Options")
    for opt_id in DEFAULT_LETTER_ORDER:
        opt_id = opt_id.upper()
        opt_tag = ET.SubElement(options_tag, opt_id)
        opt_tag.text = options_filling[opt_id]

    ET.indent(xml_root)
    xml_query = ET.tostring(xml_root, encoding='unicode')
    return xml_query

class InputFormat(Enum):
    BASE = "base"
    JSON = "json"
    XML = "xml"

class OutputFormat(Enum):
    BASE = BASE_OUTPUT_FORMAT
    JSON_ANSWER = JSON_ANSWER_OUTPUT_FORMAT
    JSON_FULL = JSON_FULL_OUTPUT_FORMAT
    XML_ANSWER = XML_ANSWER_OUTPUT_FORMAT
    XML_FULL = XML_FULL_OUTPUT_FORMAT

class ShuffleMethod(Enum):
    DEFAULT = "default"
    REVERSE = "reverse"
    LONGEST_FIRST = "longest-first"
    SHORTEST_FIRST = "shortest-first"

    MOST_KANA_RATIO = "kana-most-ratio"
    FEWEST_KANA_RATIO = "kana-fewest-ratio"

    # Not use
    # GOLD_A = "gold A"          # Always Put the answer in the option A
    # GOLD_B = "gold B"
    # GOLD_C = "gold C"
    # GOLD_D = "gold D"

DEFAULT_MAPPING = {str(letter):str(letter) for letter in DEFAULT_LETTER_ORDER}

# hiragana \u3040 - \u309F
# katagana \u30A0 - \u30FF
# katagana extension \u31F0-\u31FF
kana_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF]+')

def get_option_mapping(
    option_contents,
    shuffle_method: ShuffleMethod = ShuffleMethod.DEFAULT,
    output_format: OutputFormat = OutputFormat.BASE
) -> tuple[dict[str, str], dict[str, str]]:
    '''
    Retrieve a dictionary that could map original position to new position
    Returns:
        The original option ids to the shuffled ones
    '''
    # Temp
    original_to_shuffled = DEFAULT_MAPPING
    if shuffle_method is ShuffleMethod.REVERSE:
        original_to_shuffled = {
            str(idx):str(to_idx)
            for idx, to_idx in zip(DEFAULT_LETTER_ORDER, np.flip(DEFAULT_LETTER_ORDER))
        }
    elif shuffle_method in [ShuffleMethod.LONGEST_FIRST, ShuffleMethod.SHORTEST_FIRST]:
        sorted_source = np.argsort(-np.char.str_len(option_contents))

        # For SHORTEST_FIRST
        if shuffle_method is ShuffleMethod.SHORTEST_FIRST:
            sorted_source = sorted_source[::-1]

        original_to_shuffled = {
            str(DEFAULT_LETTER_ORDER[source_idx]):str(DEFAULT_LETTER_ORDER[pos_idx])
            for pos_idx, source_idx in enumerate(sorted_source)
        }
    elif shuffle_method in [ShuffleMethod.MOST_KANA_RATIO, ShuffleMethod.FEWEST_KANA_RATIO]:
        kana_counts = np.array([sum(len(kanas) for kanas in kana_pattern.findall(option_content)) for option_content in option_contents])
        options_len = np.char.str_len(option_contents)
        kana_ratio = kana_counts / (options_len + 1e-5)

        # Sort by ratio of kana and the total length
        # Higher Kana ratio > smaller length
        sorted_keys = (options_len, -kana_ratio)
        sorted_source = np.lexsort(sorted_keys)

        # For FEWEST_KANA_RATIO
        if shuffle_method is ShuffleMethod.FEWEST_KANA_RATIO:
            sorted_source = sorted_source[::-1]

        original_to_shuffled = {
            str(DEFAULT_LETTER_ORDER[source_idx]):str(DEFAULT_LETTER_ORDER[pos_idx])
            for pos_idx, source_idx in enumerate(sorted_source)
        }
    return original_to_shuffled


# Base Prompt
def get_query_shuffle_pair(
    curr_question : str,
    input_format_type: InputFormat = InputFormat.BASE,
    output_format_type: OutputFormat = OutputFormat.BASE,
    shuffle_method: ShuffleMethod = ShuffleMethod.DEFAULT,
):
    """Get shuffled query and its shuffle mapping
    """
    output_format = output_format_type.value
    options = np.array([curr_question[opt] for opt in DEFAULT_LETTER_ORDER])
    original_to_shuffled = get_option_mapping(options, shuffle_method)

    common_query_filling = {'Question': curr_question['Question'].strip(), "Output_format": output_format}

    # {shuffled_pos: content}
    options_filling = {original_to_shuffled[opt]: curr_question[opt].strip() for opt in DEFAULT_LETTER_ORDER}

    # Ensure the order. Sort to A -> D
    options_filling = dict(sorted(options_filling.items()))
    if input_format_type is InputFormat.BASE:
        common_query_filling.update(options_filling)
        query = BASE_TEMPLATE_FREE
        result =  (query.format_map(common_query_filling), original_to_shuffled)
    elif input_format_type is InputFormat.JSON:
        query_dict = get_json_input(common_query_filling, options_filling)
        json_query = json.dumps(query_dict, ensure_ascii=False, indent=4)
        result =  (json_query, original_to_shuffled)
    elif input_format_type is InputFormat.XML:
        xml_query = get_xml_input(common_query_filling, options_filling)
        result = (xml_query, original_to_shuffled)
    return result

def get_current_queries(
    mmmlu_subset: Dataset,
    curr_language: MMMLULanguage,
    chosen_subtasks: Union[list[str], tuple[str]],
    input_format: InputFormat = InputFormat.BASE,
    output_format: OutputFormat = OutputFormat.BASE,
    shuffle_method: ShuffleMethod = ShuffleMethod.DEFAULT
) -> list[dict]:
    '''Generate all the queries in the current setting
    '''
    query_list = []
    for task_i, curr_subtask in enumerate(chosen_subtasks):
        curr_ds = mmmlu_subset.filter(lambda x: x['Subject'] == curr_subtask)
        for curr_data in curr_ds:
            curr_query, curr_orig_to_shuffled = get_query_shuffle_pair(curr_data,
                                                                       input_format_type=input_format,
                                                                       output_format_type=output_format,
                                                                       shuffle_method=shuffle_method,
                                                                       )
            orig_ans = curr_data['Answer']
            curr_ans = curr_orig_to_shuffled[orig_ans] # Get the new position of answer
            query_dict = {
                "Question id in subtask": curr_data['Question id in subtask'],
                "Shuffle method": shuffle_method.value,
                "Original to shuffled": curr_orig_to_shuffled,
                "Input format": input_format.value,
                "Output format": output_format.value,
                "Language": curr_language,
                "Subtask": curr_subtask,
                "Query": curr_query,
                "Original correct answer": orig_ans,
                "Shuffled correct answer": curr_ans
            }
            query_list.append(query_dict)
    return query_list