

def convert_example_test(example, tokenizer, max_seq_length=512, do_evalute=False):
    result = []
    encoded_inputs = tokenizer(text=example, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    result += [input_ids, token_type_ids]
    return result

    
def convert_example_recall_infer(example,
                    tokenizer,
                    max_seq_length=512,
                    pad_to_max_seq_len=False):
    """
    Builds model inputs from a sequence.
        
    A BERT sequence has the following format:
    - single sequence: ``[CLS] X [SEP]``
    Args:
        example(obj:`list(str)`): The list of text to be converted to ids.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
    Returns:
        input_ids(obj:`list[int]`): The list of query token ids.
        token_type_ids(obj: `list[int]`): List of query sequence pair mask.
    """

    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(
            text=text,
            max_seq_len=max_seq_length,
            pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result


def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 3:
                continue
            yield {'query': data[0], 'title': data[1]}

def convert_pairwise_example(example,
                             tokenizer,
                             max_seq_length=512,
                             phase="train"):

    if phase == "train":
        query, pos_title, neg_title = example["query"], example[
            "title"], example["neg_title"]

        pos_inputs = tokenizer(
            text=query, text_pair=pos_title, max_seq_len=max_seq_length)
        neg_inputs = tokenizer(
            text=query, text_pair=neg_title, max_seq_len=max_seq_length)

        pos_input_ids = pos_inputs["input_ids"]
        pos_token_type_ids = pos_inputs["token_type_ids"]
        neg_input_ids = neg_inputs["input_ids"]
        neg_token_type_ids = neg_inputs["token_type_ids"]

        return (pos_input_ids, pos_token_type_ids, neg_input_ids,
                neg_token_type_ids)

    else:
        query, title = example["query"], example["title"]

        inputs = tokenizer(
            text=query, text_pair=title, max_seq_len=max_seq_length)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        if phase == "eval":
            return input_ids, token_type_ids, example["label"]
        elif phase == "predict":
            return input_ids, token_type_ids
        else:
            raise ValueError("not supported phase:{}".format(phase))