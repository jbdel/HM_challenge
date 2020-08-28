import numpy as np 

from transformers import BertTokenizer

def bert_encoder(seq, 
                pretrained_model_name='bert-base-uncased',
                max_seq_length=128):
    """ Sequence encoder using a pretrained Bert Tokenizer from the transformers library.

        Note: This only encodes a sequence, it is not used for training (no masks)

        Note: setting max_seq_length=128 is very restrictive. The longest sentence in the train dataset has length 433!

        Params:
            - seq: sequence to be tokenized - format: str
            - pretrained_model_name: name of the pretrained model (cf library) - format: str
            - max_seq_length: maximum length of encoded sequence - format: int
        
        Returns: a dictionnary with keys/values:
            - "input_tokens": tokens for the sequence - format: list[str]
            - "input_ids": token ids for the sequence - format: np.array of shape (max_seq_length,) and type np.uint64
            - "segment_ids": segment ids or token type ids for the sequence - format: same as input_ids, but type np.uint8
            - "input_mask": attention mask for the sequence - format: same as segment_ids
    """
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, do_lower_case=True)

    encoded_seq = tokenizer(
        seq,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    output = {}
    output["input_tokens"] = tokenizer.tokenize(seq)
    output["input_ids"] = np.array(encoded_seq["input_ids"], dtype=np.uint64)
    output["segment_ids"] = np.array(encoded_seq["token_type_ids"], dtype=np.uint8)      
    output["input_mask"] = np.array(encoded_seq["attention_mask"], dtype=np.uint8)

    return output


