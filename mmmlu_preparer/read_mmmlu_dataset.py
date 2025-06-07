import numpy as np
from enum import Enum
from typing import Union
import datasets
from datasets import load_dataset, Value, Dataset

DEFAULT_LETTER_ORDER = np.array(["A", "B", "C", "D"])

# Manually select. Prefer professional over high school
TARGET_SUBTASKS = ('abstract_algebra', 'virology', 'astronomy', 'marketing', 'college_biology', 'college_chemistry', 'machine_learning',
    'econometrics', 'electrical_engineering', 'philosophy', 'global_facts', 'prehistory',
    'high_school_geography', 'security_studies', 'professional_psychology', 'sociology', 'jurisprudence'
)

# https://github.com/hendrycks/test/blob/master/categories.py
SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

class MMMLULanguage(Enum):
    EN = "EN"
    JA_JP = "JA_JP"


def to_mmmlu_format(mmlu_dataset: Dataset):
    """
    Convert the format of English mmlu to mmmlu.
    """
    for col_name in mmlu_dataset.features:
        mmlu_dataset = mmlu_dataset.rename_column(col_name, col_name.capitalize())

    # Add options columns
    choice_arr = np.array(mmlu_dataset['Choices'])
    for idx, letter in enumerate(DEFAULT_LETTER_ORDER):
        mmlu_dataset = mmlu_dataset.add_column(letter, choice_arr[:, idx])
    mmlu_dataset = mmlu_dataset.remove_columns('Choices')

    # From number answer cols to letter answer cols
    mmlu_dataset = mmlu_dataset.cast_column('Answer', Value('string'))
    mmlu_dataset = mmlu_dataset.map(lambda x: {"Answer": DEFAULT_LETTER_ORDER[int(x['Answer'])]})

    # Add Question id in subtask for convenience
    df = mmlu_dataset.to_pandas()
    df['Question id in subtask'] = df.groupby("Subject").cumcount()
    columns = list(df.columns)
    question_id_col_name = columns.pop()
    columns.insert(0, question_id_col_name)
    df = df[columns]
    mmlu_dataset = Dataset.from_pandas(df)
    return mmlu_dataset

def create_mmmlu_dataset(language: MMMLULanguage = MMMLULanguage.EN):
    '''Huggingface dataset of MMMLU
    '''
    # [En, JA_JP]
    if language is MMMLULanguage.EN:
        mmmlu_ds = load_dataset('cais/mmlu', 'all', split='test')
        mmmlu_ds = to_mmmlu_format(mmmlu_ds)
    else:
        mmmlu_ds = load_dataset("openai/MMMLU", language.value, split='test')
        mmmlu_ds = mmmlu_ds.rename_column('Unnamed: 0', "Question id in subtask")
    return mmmlu_ds


def get_subtasks():
    '''Get 17 subtasks and their subcategories
    '''
    subtasks = {}
    for subtask, task_subcategory in SUBCATEGORIES.items():
        curr_category = task_subcategory[0]
        if curr_category not in subtasks:
            subtasks[curr_category] = []
        subtasks[curr_category].append(subtask)

    return subtasks

datasets.disable_progress_bar()
def sample_first_n_data_from_subtask(ds: Dataset, target_subtasks: Union[list[str], tuple[str]], sample_first_n: int = 100) -> Dataset:
    '''Sample first n data in each taget subtask
    '''
    subtask_set = set(target_subtasks)
    sampled_ds = ds.filter(lambda x: x['Subject'] in subtask_set and x['Question id in subtask'] < sample_first_n)
    return sampled_ds